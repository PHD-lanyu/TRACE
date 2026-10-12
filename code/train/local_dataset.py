import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import Dataset


DEFAULT_SYSTEM_PROMPT = "你是一个严谨的助手。请直接给出最终答案，避免无关内容。"
DEFAULT_SYSTEM_PROMPT_EN = "You are a careful assistant. Provide the final answer directly, without extra chatter."

def _detect_prompt_style(q: str) -> str:
    """Heuristically detect whether a prompt is Chinese-like or English-like.
    Returns "cn" or "en".
    """
    s = (q or "").lstrip()
    low = s.lower()
    if low.startswith("question:") or "retrieved context" in low or "options" in low:
        return "en"
    if s.startswith("问题") or "参考资料" in s or "选项" in s:
        return "cn"
    ascii_cnt = sum(1 for ch in s[:400] if ord(ch) < 128)
    total_cnt = max(1, min(len(s), 400))
    return "en" if (ascii_cnt / total_cnt) > 0.75 else "cn"

def _contains_any(s: str, needles: Iterable[str]) -> bool:
    low = (s or "").lower()
    return any(n.lower() in low for n in needles)



def _read_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # common containers
        for k in ("data", "examples", "samples", "items"):
            if k in obj and isinstance(obj[k], list):
                return obj[k]
    raise ValueError(f"Unsupported JSON schema in {path}: expecting a list or a dict containing a list.")


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Invalid JSONL at line {ln} in {path}: {e}")
    return records


def load_records(path: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        return _read_jsonl(path)
    if ext == ".json":
        return _read_json(path)
    # allow "unknown" extension but JSONL-like content
    try:
        return _read_json(path)
    except Exception:
        return _read_jsonl(path)


def _first_present(d: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _normalize_context(ctx: Any) -> Optional[str]:
    if ctx is None:
        return None
    if isinstance(ctx, str):
        s = ctx.strip()
        return s if s else None
    if isinstance(ctx, list):
        parts: List[str] = []
        for x in ctx:
            if x is None:
                continue
            if isinstance(x, str) and x.strip():
                parts.append(x.strip())
            elif isinstance(x, dict):
                # try common dict keys
                val = _first_present(x, ["text", "content", "passage", "document"])  # type: ignore
                if isinstance(val, str) and val.strip():
                    parts.append(val.strip())
        joined = "\n\n".join(parts).strip()
        return joined if joined else None
    if isinstance(ctx, dict):
        val = _first_present(ctx, ["text", "content", "passage", "document"])  # type: ignore
        if isinstance(val, str) and val.strip():
            return val.strip()
    return str(ctx).strip() or None


def _infer_qa(record: Dict[str, Any]) -> Tuple[str, str, Optional[str]]:
    """Infer (question, answer, context) from a flexible record schema."""
    question = _first_present(record, ["question", "question_text", "query", "prompt", "instruction", "input_text", "input", "user", "user_prompt"])  # type: ignore
    if question is None:
        raise KeyError("Cannot infer question. Expected one of: question/query/prompt/instruction/input_text")
    if not isinstance(question, str):
        question = str(question)
    # For instruction-tuning style: instruction + optional input
    if "instruction" in record and isinstance(record.get("instruction"), str):
        inst = record.get("instruction", "")
        inp = record.get("input", "")
        if isinstance(inp, str) and inp.strip():
            question = f"{inst.strip()}\n\n输入：{inp.strip()}"
        else:
            question = inst.strip()

    answer = _first_present(record, ["answer", "final", "final_answer", "output", "response", "completion", "target", "assistant", "assistant_response"])  # type: ignore
    if answer is None:
        raise KeyError("Cannot infer answer. Expected one of: answer/output/response/completion/target")
    if not isinstance(answer, str):
        answer = str(answer)

    # context: accept single field or plural fields
    ctx = _first_present(
        record,
        [
            "context",
            "contexts",
            "retrieved_context",
            "retrieved",
            "retrieval",
            "evidence",
            "ctx",
            "context_text",
            "passage",
            "passages",
            "docs",
            "documents",
            "golden_context",
            "negative_context",
        ],
    )
    context = _normalize_context(ctx)
    return question.strip(), answer.strip(), context


def _apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    """Apply chat template if available; otherwise, fall back to a simple format."""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            pass

    # fallback: classic prompt
    sys = ""
    if messages and messages[0]["role"] == "system":
        sys = f"系统：{messages[0]['content']}\n\n"
        msgs = messages[1:]
    else:
        msgs = messages

    user = ""
    assistant = ""
    for m in msgs:
        if m["role"] == "user":
            user = m["content"]
        elif m["role"] == "assistant":
            assistant = m["content"]
    return f"{sys}下面列出了一个问题。请写出问题的答案。\n### 问题:\n{user}\n### 答案:\n{assistant}"


class LocalJsonDataset:
    def __init__(
        self,
        json_file: str,
        tokenizer,
        max_seq_length: int = 4096,
        system_prompt: Optional[str] = None,
        use_context_label: bool = False,
        max_samples: Optional[int] = None,
        prompt_style: str = "auto",
    ):
        self.json_file = json_file
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.system_prompt = (system_prompt or DEFAULT_SYSTEM_PROMPT).strip()
        self.use_context_label = use_context_label
        self.max_samples = max_samples
        self.prompt_style = (prompt_style or "auto").strip().lower()
        self.dataset = self._build_dataset()

    def _record_to_text(self, record: Dict[str, Any]) -> str:
        # if pre-tokenized text provided
        txt = record.get("text")
        if isinstance(txt, str) and txt.strip():
            out = txt.strip()
        # chat messages schema
        elif isinstance(record.get("messages"), list):
            messages = record["messages"]
            # ensure system prompt exists
            if self.system_prompt:
                if not messages or messages[0].get("role") != "system":
                    messages = [{"role": "system", "content": self.system_prompt}] + messages
            out = _apply_chat_template(self.tokenizer, messages)
        else:
            q, a, context = _infer_qa(record)
            notice = ""
            if self.use_context_label:
                label = _first_present(record, ["context_type", "context_label", "label", "is_negative"])  # type: ignore
                if isinstance(label, str) and label.lower() in {"negative", "neg", "bad", "false"}:
                    notice = "注意：以下参考资料可能包含错误或误导信息。\nNOTE: The following retrieved context may contain errors or misleading information.\n\n"
                elif label is True:
                    notice = "注意：以下参考资料可能包含错误或误导信息。\nNOTE: The following retrieved context may contain errors or misleading information.\n\n"
            # Decide prompt language / prefix style.
            style = self.prompt_style
            if style not in {"auto", "cn", "en"}:
                style = "auto"
            if style == "auto":
                style = _detect_prompt_style(q)

            # If the question already contains common prefixes, do not wrap it again.
            q_has_context_mark = _contains_any(q, ["参考资料", "retrieved context", "context:", "evidence:"])
            q_has_question_mark = q.lstrip().lower().startswith("question:") or q.lstrip().startswith("问题")
            base_user = q.strip() if (q_has_question_mark and (context is None or q_has_context_mark)) else None

            if base_user is None:
                if style == "en":
                    if context:
                        user = (
                            f"{notice}Question:\n{q}\n\n"
                            f"Retrieved context (may be empty / may be wrong):\n{context}\n\n"
                            "Please answer the question. If the retrieved context conflicts with reliable knowledge, do not blindly follow it."
                        )
                    else:
                        user = (
                            f"{notice}Question:\n{q}\n\n"
                            "Please answer the question."
                        )
                else:
                    if context:
                        user = f"{notice}问题：{q}\n\n参考资料：\n{context}\n\n请基于问题与参考资料给出最终答案。"
                    else:
                        user = f"{notice}问题：{q}\n\n请给出最终答案。"
            else:
                # question already contains the desired structure/prefixes
                if context and (not q_has_context_mark):
                    if style == "en":
                        user = (
                            f"{notice}{base_user}\n\n"
                            f"Retrieved context (may be empty / may be wrong):\n{context}\n\n"
                            "Please answer the question."
                        )
                    else:
                        user = f"{notice}{base_user}\n\n参考资料：\n{context}\n\n请给出最终答案。"
                else:
                    user = f"{notice}{base_user}"

            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": a})
            out = _apply_chat_template(self.tokenizer, messages)

        # add EOS if needed
        eos = getattr(self.tokenizer, "eos_token", None)
        if isinstance(eos, str) and eos and not out.endswith(eos):
            out = out + eos
        return out

    def _build_dataset(self) -> Dataset:
        records = load_records(self.json_file)
        if self.max_samples is not None:
            records = records[: int(self.max_samples)]
        texts: List[str] = []
        for rec in records:
            if not isinstance(rec, dict):
                rec = {"text": str(rec)}
            texts.append(self._record_to_text(rec))
        return Dataset.from_dict({"text": texts})

    def get_dataset(self) -> Dataset:
        return self.dataset
