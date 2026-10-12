import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib import error, request

import backoff

from .openai_utils import AccessTerminatedException, OutOfQuotaException
from .openai_utils import model2max_context, num_tokens_from_string

# OpenAI is optional.
try:
    import openai  # type: ignore
    from openai.error import APIConnectionError, APIError, RateLimitError, ServiceUnavailableError  # type: ignore

    _OPENAI_AVAILABLE = True
except Exception:  # pragma: no cover
    openai = None
    RateLimitError = APIError = ServiceUnavailableError = APIConnectionError = Exception
    _OPENAI_AVAILABLE = False

# Transformers is optional and only needed for local HuggingFace models.
try:  # pragma: no cover - import availability depends on runtime
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _HF_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    _HF_AVAILABLE = False


support_models = ["gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-4", "gpt-4-0314"]


_HF_CACHE: Dict[Tuple[str, str, str], Tuple[object, object]] = {}


def _env_truthy(key: str, default: str = "0") -> bool:
    v = os.getenv(key, default).strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _env_to_bool_or_none(key: str):
    v = os.getenv(key, "").strip().lower()
    if not v:
        return None
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _looks_like_local_hf_model(model_name: str) -> bool:
    if not model_name:
        return False
    p = Path(model_name)
    if p.exists():
        return True
    # Also allow explicit scheme-like prefixes.
    return model_name.startswith("hf://") or model_name.startswith("local://")


class Agent:
    def __init__(self, model_name: str, name: str, temperature: float, sleep_time: float = 0) -> None:
        self.model_name = model_name
        self.name = name
        self.temperature = temperature
        self.memory_lst = []
        self.sleep_time = sleep_time
        self.verbose = _env_truthy("MAD_VERBOSE", "1")
        self.last_thinking: str = ""
        self.ollama_api_key: str = ""
        self.ollama_base_url: str = ""
        self.hf_device: str = os.getenv("HF_DEVICE", "cuda:0" if (torch is not None and torch.cuda.is_available()) else "cpu")
        self.hf_dtype: str = os.getenv("HF_DTYPE", "bfloat16")

    # ------------------------- Provider selection -------------------------
    def _use_openai(self) -> bool:
        provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        if provider in {"hf", "huggingface", "transformers", "local_hf"}:
            return False
        if provider in {"ollama", "local"}:
            return False
        if provider in {"openai"}:
            return True
        if _looks_like_local_hf_model(self.model_name):
            return False
        return self.model_name in support_models

    def _use_hf_local(self) -> bool:
        provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        if provider in {"hf", "huggingface", "transformers", "local_hf"}:
            return True
        return _looks_like_local_hf_model(self.model_name)

    # ------------------------- OpenAI backend -------------------------
    def _query_openai(self, messages: "list[dict]", max_tokens: int, api_key: str, temperature: float) -> str:
        if not _OPENAI_AVAILABLE:
            raise RuntimeError(
                "openai package is not available, but LLM_PROVIDER/openai was selected. "
                "Install openai or switch to Ollama/local HF."
            )

        last_err = None
        for attempt in range(1, 21):
            try:
                time.sleep(self.sleep_time)
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=api_key,
                )
                return response["choices"][0]["message"]["content"]
            except RateLimitError as e:
                last_err = e
                user_msg = getattr(e, "user_message", "") or str(e)
                if "You exceeded your current quota" in user_msg:
                    raise OutOfQuotaException(api_key)
                if "Your access was terminated" in user_msg:
                    raise AccessTerminatedException(api_key)
            except (APIError, ServiceUnavailableError, APIConnectionError) as e:
                last_err = e

            sleep_s = min(60.0, (2 ** min(attempt, 6)) + random.random())
            time.sleep(sleep_s)

        raise last_err if last_err else RuntimeError("OpenAI query failed")

    # ------------------------- Ollama backend -------------------------
    def _ollama_chat(self, messages: "list[dict]", num_predict: int, temperature: float, think=None) -> str:
        base_url = (self.ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        endpoint = f"{base_url}/api/chat"
        timeout_s = float(os.getenv("OLLAMA_TIMEOUT", "60000"))

        options: dict = {"temperature": temperature}
        if num_predict and num_predict > 0:
            options["num_predict"] = int(num_predict)

        # IMPORTANT for Ollama stability: always pass an explicit runtime context.
        # If this is omitted, Ollama may use the model/server default, which can be
        # very large (e.g. 131072) and create a huge KV cache on GPU.
        num_ctx_env = os.getenv("OLLAMA_NUM_CTX", os.getenv("OLLAMA_CONTEXT_LENGTH", "8192")).strip()
        if num_ctx_env:
            try:
                options["num_ctx"] = int(num_ctx_env)
            except ValueError:
                pass

        if think is None:
            think = _env_to_bool_or_none("OLLAMA_THINK")

        def _do_request(req_payload: dict) -> dict:
            data = json.dumps(req_payload, ensure_ascii=False).encode("utf-8")
            headers = {"Content-Type": "application/json"}

            key = (os.getenv("OLLAMA_API_KEY", "") or "").strip()
            if not key:
                key = (getattr(self, "ollama_api_key", "") or "").strip()
            if not key:
                key = (getattr(self, "openai_api_key", "") or "").strip()
            if key:
                headers["Authorization"] = f"Bearer {key}"

            req = request.Request(endpoint, data=data, headers=headers, method="POST")
            with request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read().decode("utf-8")
            obj = json.loads(body)

            if isinstance(obj, dict) and obj.get("error"):
                raise RuntimeError(f"Ollama error: {obj['error']}")
            if not isinstance(obj, dict):
                raise RuntimeError(f"Unexpected Ollama response format: {obj}")
            return obj

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        # Keep only one large Ollama model resident unless explicitly configured.
        # "0" means unload right after the request; this avoids Qwen3:32b and
        # deepseek-r1:8b lingering together and fighting for 3090 VRAM.
        keep_alive_env = os.getenv("OLLAMA_KEEP_ALIVE", "24h").strip()
        if keep_alive_env:
            payload["keep_alive"] = keep_alive_env

        if _env_truthy("PRINT_OLLAMA_PAYLOAD", "0"):
            print(f"[Ollama payload] model={self.model_name} options={options} keep_alive={payload.get('keep_alive')}")

        if think is not None:
            payload["think"] = think

        obj1 = _do_request(payload)
        msg1 = (obj1.get("message") or {}) if isinstance(obj1, dict) else {}
        thinking1 = msg1.get("thinking") if isinstance(msg1.get("thinking"), str) else ""
        content1 = msg1.get("content")
        if not isinstance(content1, str):
            raise RuntimeError(f"Unexpected Ollama response format: {obj1}")

        self.last_thinking = thinking1
        if _env_truthy("PRINT_OLLAMA_THINKING", "0") and self.last_thinking.strip():
            print(f"----- {self.name} (thinking trace) -----\n{self.last_thinking}\n")

        if content1.strip() == "" and thinking1.strip() != "" and _env_truthy("OLLAMA_FALLBACK_NO_THINK", "1"):
            if think is None or think is True:
                payload2 = dict(payload)
                payload2["think"] = False
                obj2 = _do_request(payload2)
                msg2 = (obj2.get("message") or {}) if isinstance(obj2, dict) else {}
                content2 = msg2.get("content")
                if isinstance(content2, str) and content2.strip():
                    return content2
                thinking2 = msg2.get("thinking")
                if not self.last_thinking.strip() and isinstance(thinking2, str):
                    self.last_thinking = thinking2

        if content1.strip() == "":
            if _env_truthy("OLLAMA_EMPTY_RETURNS_THINKING", "0") and self.last_thinking.strip():
                return self.last_thinking
            if _env_truthy("OLLAMA_EMPTY_IS_ERROR", "1"):
                raise RuntimeError(f"Ollama returned empty message.content (thinking_len={len(self.last_thinking)}).")

        return content1

    @backoff.on_exception(
        backoff.expo,
        (error.URLError, error.HTTPError, TimeoutError, ConnectionError, RuntimeError),
        max_tries=8,
    )
    def _query_ollama_with_retry(self, messages: "list[dict]", num_predict: int, temperature: float, think=None) -> str:
        time.sleep(self.sleep_time)
        return self._ollama_chat(messages=messages, num_predict=num_predict, temperature=temperature, think=think)

    # ------------------------- Local HF backend -------------------------
    def _resolve_hf_path(self) -> str:
        if self.model_name.startswith("hf://"):
            return self.model_name[len("hf://") :]
        if self.model_name.startswith("local://"):
            return self.model_name[len("local://") :]
        return self.model_name

    def _torch_dtype(self):
        if torch is None:
            return None
        name = str(self.hf_dtype or "").lower()
        if name in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if name in {"fp16", "float16", "half"}:
            return torch.float16
        if name in {"fp32", "float32"}:
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def _load_hf_model_and_tokenizer(self):
        if not _HF_AVAILABLE:
            raise RuntimeError(
                "transformers/torch are not available, but a local HuggingFace model path was requested."
            )

        model_path = self._resolve_hf_path()
        device = str(getattr(self, "hf_device", "cpu") or "cpu")
        dtype_name = str(self.hf_dtype or "auto")
        cache_key = (model_path, device, dtype_name)
        if cache_key in _HF_CACHE:
            return _HF_CACHE[cache_key]

        torch_dtype = self._torch_dtype()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        device_map = None
        if device.startswith("cuda"):
            # Recommended usage: run python with CUDA_VISIBLE_DEVICES pinned to the intended GPU,
            # then keep hf_device=cuda:0 inside the process.
            device_map = {"": 0}

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )

        if device_map is None:
            model.to(device)
        model.eval()

        _HF_CACHE[cache_key] = (tokenizer, model)
        return tokenizer, model

    def _query_hf_local(self, messages: "list[dict]", max_tokens: int, temperature: float) -> str:
        tokenizer, model = self._load_hf_model_and_tokenizer()
        device = str(getattr(self, "hf_device", "cpu") or "cpu")

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        if device.startswith("cuda") or device == "cpu":
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        do_sample = float(temperature) > 0
        gen_kwargs = {
            "max_new_tokens": int(max(1, max_tokens)),
            "do_sample": do_sample,
            "temperature": float(temperature) if do_sample else None,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if not do_sample:
            gen_kwargs.pop("temperature", None)

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[-1]
        generated_ids = outputs[0][input_len:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text.strip()

    # ------------------------- Public API -------------------------
    def query(self, messages: "list[dict]", max_tokens: int, api_key: str, temperature: float, think=None) -> str:
        if self._use_openai():
            assert self.model_name in support_models, f"Not support {self.model_name}. Choices: {support_models}"
            return self._query_openai(messages, max_tokens=max_tokens, api_key=api_key, temperature=temperature)

        if self._use_hf_local():
            time.sleep(self.sleep_time)
            return self._query_hf_local(messages, max_tokens=max_tokens, temperature=temperature)

        self.ollama_api_key = (api_key or "").strip()
        return self._query_ollama_with_retry(messages, num_predict=max_tokens, temperature=temperature, think=think)

    def set_meta_prompt(self, meta_prompt: str):
        self.memory_lst.append({"role": "system", "content": f"{meta_prompt}"})

    def add_event(self, event: str):
        self.memory_lst.append({"role": "user", "content": f"{event}"})

    def add_memory(self, memory: str):
        self.memory_lst.append({"role": "assistant", "content": f"{memory}"})
        if self.verbose:
            shown = memory if (memory or "").strip() else "<EMPTY>"
            print(f"----- {self.name} -----\n{shown}\n")

    def set_verbose(self, verbose: bool):
        self.verbose = bool(verbose)

    def ask(self, temperature: float = None, think=None):
        num_context_token = sum(num_tokens_from_string(m["content"], self.model_name) for m in self.memory_lst)

        if self._use_openai():
            max_ctx = model2max_context.get(self.model_name, 3900)
            max_token = max(1, max_ctx - num_context_token)
        elif self._use_hf_local():
            max_ctx = model2max_context.get(self.model_name, int(os.getenv("DEFAULT_MAX_CONTEXT", "8192")))
            max_token = int(os.getenv("HF_MAX_NEW_TOKENS", "256"))
            if num_context_token > max_ctx * 0.9:
                max_token = max(64, int(max_token * 0.5))
        else:
            max_ctx = model2max_context.get(self.model_name, int(os.getenv("DEFAULT_MAX_CONTEXT", "8192")))
            max_token = int(os.getenv("OLLAMA_NUM_PREDICT", "1024"))
            if num_context_token > max_ctx * 0.9:
                max_token = max(128, int(max_token * 0.5))

        return self.query(
            self.memory_lst,
            max_token,
            api_key=getattr(self, "openai_api_key", ""),
            temperature=temperature if temperature is not None else self.temperature,
            think=think,
        )