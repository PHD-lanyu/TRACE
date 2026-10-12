from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training

from local_dataset import LocalJsonDataset, load_records

POS_FOURWAY_IDS = {
    "external_correct_answer_correct": 0,
    "external_wrong_answer_correct": 1,
    # compatibility with older names
    "S11": 0,
    "S21": 1,
}
NEG_FOURWAY_IDS = {
    "external_correct_answer_wrong": 0,
    "external_wrong_answer_wrong": 1,
    # compatibility with older names
    "S12": 0,
    "S22": 1,
}


NEG_TYPE_TO_ID = {
    "none": 0,
    "idk": 1,
    "final": 2,
    "internal": 3,
    "meta": 4,
    # v21 candidate-aware hard negatives
    "candidate": 5,
    "context": 6,
    "drift": 7,
    # v21.3 answer-completeness / boundary near-miss negatives
    "boundary": 8,
}

TRAIN_METHOD_CONSTANTS = {
    "ignore_reason_loss": True,
    "neg_loss_on_answer_only": True,
    "ul_eps": 1e-6,
    "pos_template_ce_weight": 0.20,
    "pos_answer_ce_weight": 0.35,
    "pos_late_token_boost": 1.5,
    "pos_early_terminate_weight": 0.08,
    "source_cls_weight": 0.05,
    "source_cls_dropout": 0.1,
    "source_cls_internal_weight": 1.4,
    "source_cls_external_weight": 1.0,
    "source_cls_both_weight": 0.8,
    "source_cls_abstain_weight": 0.35,
    "source_cls_ignore_abstain": True,
    "internal_pos_ce_bonus": 1.15,
    "internal_answer_ce_bonus": 1.35,
    "internal_neg_ul_bonus": 1.25,
    "internal_idk_margin_weight": 0.0,
    "internal_idk_margin": 0.6,
    "internal_meta_margin_weight": 0.0,
    "internal_meta_margin": 0.5,
    "internal_final_margin_weight": 0.8,
    "internal_final_margin": 0.25,
    "enable_internal_ema_beta": True,
    "internal_ema_alpha": 0.05,
    "internal_ema_beta_min": 0.25,
}

def _is_bfloat16_supported() -> bool:
    return bool(torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())

# -------------------------
# Helpers
# -------------------------
def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _apply_chat_template_or_fallback(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=bool(add_generation_prompt),
            )
        except Exception:
            pass

    chunks = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        chunks.append(f"{role.upper()}: {content}")
    if add_generation_prompt:
        chunks.append("ASSISTANT:")
    return "\n\n".join(chunks)


def _infer_category_from_stage_fields(rec: Dict[str, Any]) -> Optional[str]:
    try:
        s1 = bool(rec["stage1"]["correct"])
        mode = rec["stage2"]["mode"]
        s2 = bool(rec["stage2"]["correct"])
        if s1 and mode == "negative_context":
            return "S11" if s2 else "S12"
        if (not s1) and mode == "golden_context":
            return "S21" if s2 else "S22"
        return None
    except Exception:
        return None


class SourceAwareModel(nn.Module):
    def __init__(self, base_model: nn.Module, num_labels: int = 4, dropout: float = 0.1):
        super().__init__()
        self.base_model = base_model
        self.config = getattr(base_model, "config", None)

        # NOTE:
        # Trainer checks whether a model is a *purely* quantized base model.
        # Once we wrap a LoRA/PEFT model inside a plain nn.Module, that PEFT
        # identity can become invisible to the Trainer and it may raise:
        # "You cannot perform fine-tuning on purely quantized models ..."
        #
        # We keep the underlying LoRA adapters intact, but expose explicit flags
        # on the wrapper so Trainer understands this is NOT a bare quantized
        # model that lacks trainable adapters.
        self._hf_peft_config_loaded = bool(getattr(base_model, "_hf_peft_config_loaded", True))
        self.peft_config = getattr(base_model, "peft_config", None)
        self.is_quantized = False
        self.hf_quantizer = None
        self.is_loaded_in_4bit = bool(getattr(base_model, "is_loaded_in_4bit", False))
        self.is_loaded_in_8bit = bool(getattr(base_model, "is_loaded_in_8bit", False))

        hidden_size = None
        cfg = self.config
        for attr in ("hidden_size", "d_model", "n_embd"):
            hidden_size = getattr(cfg, attr, None) if cfg is not None else None
            if hidden_size is not None:
                break
        if hidden_size is None and cfg is not None and hasattr(cfg, "text_config"):
            for attr in ("hidden_size", "d_model", "n_embd"):
                hidden_size = getattr(cfg.text_config, attr, None)
                if hidden_size is not None:
                    break
        if hidden_size is None:
            raise ValueError("Cannot infer hidden size for source classification head.")

        self.source_dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()
        self.source_classifier = nn.Linear(int(hidden_size), int(num_labels))

    def forward(self, *args, **kwargs):
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        out = self.base_model(*args, **kwargs)
        hidden = out.hidden_states[-1]
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is None:
            pooled = hidden[:, -1, :]
        else:
            idx = attention_mask.long().sum(dim=1).clamp_min(1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]

        if self.source_classifier.weight.device != pooled.device:
            self.source_classifier = self.source_classifier.to(pooled.device)
        pooled = self.source_dropout(pooled)
        if pooled.dtype != self.source_classifier.weight.dtype:
            pooled = pooled.to(self.source_classifier.weight.dtype)

        out.source_logits = self.source_classifier(pooled)
        return out

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.base_model.set_input_embeddings(value)

    def gradient_checkpointing_enable(self, *args, **kwargs):
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            return self.base_model.gradient_checkpointing_enable(*args, **kwargs)
        return None

    def gradient_checkpointing_disable(self):
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            return self.base_model.gradient_checkpointing_disable()
        return None

    def prepare_inputs_for_generation(self, *args, **kwargs):
        if hasattr(self.base_model, "prepare_inputs_for_generation"):
            return self.base_model.prepare_inputs_for_generation(*args, **kwargs)
        raise AttributeError("Underlying model does not implement prepare_inputs_for_generation")

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


# -------------------------
# Collator
# -------------------------
class DebateULDataCollator:
    """Collate batches for positive-CE / negative-UL training.

    Expected record schema:
    {
      "messages": [...],
      "pos_answer": "...",   # optional
      "neg_answer": "...",   # optional; should correspond to a wrong final answer, not merely wrong retrieved context
      "category": "external_correct_answer_correct|...",
      "sample_weight": 1.0,
    }
    """

    def __init__(
        self,
        tokenizer,
        max_seq_length: int = 4096,
        eos_add: bool = True,
        track_fourway_category: bool = False,
        ignore_reason_loss: bool = True,
        neg_loss_on_answer_only: bool = True,
    ):
        self.tok = tokenizer
        self.max_len = int(max_seq_length)
        self.eos_add = bool(eos_add)
        self.eos = getattr(tokenizer, "eos_token", None) or ""
        self.track_fourway_category = bool(track_fourway_category)
        self.ignore_reason_loss = bool(ignore_reason_loss)
        self.neg_loss_on_answer_only = bool(neg_loss_on_answer_only)

    @staticmethod
    def _truncate_answer_before_reason(answer: str) -> str:
        s = str(answer or "")
        patterns = [
            r"\nReason\s*[:：]",
            r"\nRationale\s*[:：]",
            r"\n理由\s*[:：]",
            r"\n原因\s*[:：]",
        ]
        cut = None
        for pat in patterns:
            m = re.search(pat, s, flags=re.IGNORECASE)
            if m is not None:
                cut = m.start() if cut is None else min(cut, m.start())
        if cut is None:
            return s
        return s[:cut].rstrip()

    @staticmethod
    def _split_answer_template(answer: str) -> Tuple[str, str, str]:
        """Split a templated assistant target into prefix / answer_core / suffix.

        For negative UL we only want to penalize the answer content, not the shared
        template tokens like <think> / <answer>. Otherwise the model is explicitly
        discouraged from emitting the evaluation-time tags.
        """
        s = str(answer or "")
        m = re.search(r"(?is)^(.*?<answer>\s*)(.*?)(\s*</answer>.*)$", s)
        if m is None:
            return "", s, ""
        return m.group(1), m.group(2), m.group(3)

    def _encode_one(self, messages: List[Dict[str, str]], answer: str, answer_loss_mode: str = "full") -> Dict[str, Any]:
        prompt_msgs = [m for m in messages if m.get("role") != "assistant"]
        prompt_text = _apply_chat_template_or_fallback(self.tok, prompt_msgs, add_generation_prompt=True)

        raw_answer = answer or ""
        supervised_answer = self._truncate_answer_before_reason(raw_answer) if self.ignore_reason_loss else raw_answer
        full_text = prompt_text + supervised_answer
        if self.eos_add and self.eos and (not full_text.endswith(self.eos)):
            full_text = full_text + self.eos

        full = self.tok(
            full_text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_attention_mask=True,
        )
        prompt_ids = self.tok(
            prompt_text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]
        prompt_len = len(prompt_ids)

        input_ids = torch.tensor(full["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(full["attention_mask"], dtype=torch.long)
        labels = torch.full_like(input_ids, -100)
        answer_mask = torch.zeros_like(input_ids)

        prefix_text, answer_core, _suffix_text = self._split_answer_template(supervised_answer)
        if answer_core:
            start_text = prompt_text + prefix_text
            end_text = prompt_text + prefix_text + answer_core
            start_ids = self.tok(
                start_text,
                truncation=True,
                max_length=self.max_len,
                padding=False,
                return_attention_mask=False,
            )["input_ids"]
            end_ids = self.tok(
                end_text,
                truncation=True,
                max_length=self.max_len,
                padding=False,
                return_attention_mask=False,
            )["input_ids"]
            start_idx = min(len(start_ids), answer_mask.numel())
            end_idx = min(len(end_ids), answer_mask.numel())
            if end_idx > start_idx:
                answer_mask[start_idx:end_idx] = 1

        if answer_loss_mode == "answer_only":
            if answer_mask.any():
                labels[answer_mask.bool()] = input_ids[answer_mask.bool()]
        else:
            start_idx = min(prompt_len, labels.numel())
            labels[start_idx:] = input_ids[start_idx:]

        labels[attention_mask == 0] = -100
        answer_mask[attention_mask == 0] = 0
        template_mask = ((labels != -100) & (answer_mask == 0)).long()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "answer_mask": answer_mask,
            "template_mask": template_mask,
        }

    def _pad_stack(self, encs: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pad_id = getattr(self.tok, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tok, "eos_token_id", 0)

        max_len = max(int(e["input_ids"].numel()) for e in encs)
        bsz = len(encs)
        input_ids = torch.full((bsz, max_len), int(pad_id), dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        labels = torch.full((bsz, max_len), -100, dtype=torch.long)
        answer_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        template_mask = torch.zeros((bsz, max_len), dtype=torch.long)

        for i, e in enumerate(encs):
            L = int(e["input_ids"].numel())
            input_ids[i, :L] = e["input_ids"]
            attention_mask[i, :L] = e["attention_mask"]
            labels[i, :L] = e["labels"]
            if "answer_mask" in e:
                answer_mask[i, :L] = e["answer_mask"]
            if "template_mask" in e:
                template_mask[i, :L] = e["template_mask"]
        return input_ids, attention_mask, labels, answer_mask, template_mask

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pos_items, neg_items = [], []
        pair_pos_items = []  # v21: aligned positives for records that also have a negative answer
        pos_w, neg_w = [], []
        pos_cat, neg_cat = [], []
        pos_source, neg_source = [], []
        pos_neg_type, neg_neg_type = [], []

        for f in features:
            messages = f.get("messages")
            if not isinstance(messages, list):
                q = f.get("question") or f.get("prompt") or ""
                sys = f.get("system_prompt") or ""
                messages = []
                if sys:
                    messages.append({"role": "system", "content": sys})
                messages.append({"role": "user", "content": q})

            w = float(f.get("sample_weight", 1.0))
            category = str(f.get("category", "")).strip()
            source_label_id = f.get("source_label_id", f.get("preferred_source_id", -1))
            try:
                source_label_id = int(source_label_id)
            except Exception:
                source_label_id = -1

            neg_type = str(f.get("neg_type", f.get("negative_type", "none")) or "none").strip().lower()
            neg_type_id = NEG_TYPE_TO_ID.get(neg_type, 0)
            pos_ans = (f.get("pos_answer") or "").strip()
            neg_ans = (f.get("neg_answer") or "").strip()

            if pos_ans:
                pos_items.append(self._encode_one(messages, pos_ans, answer_loss_mode="full"))
                pos_w.append(w)
                pos_source.append(source_label_id)
                pos_neg_type.append(neg_type_id)
                if self.track_fourway_category:
                    pos_cat.append(POS_FOURWAY_IDS.get(category, -1))
            if neg_ans:
                neg_mode = "answer_only" if self.neg_loss_on_answer_only else "full"
                neg_items.append(self._encode_one(messages, neg_ans, answer_loss_mode=neg_mode))
                # Keep an aligned positive encoding for pairwise answer-margin loss.
                # This avoids pairing negatives with unrelated pos-only records in mixed batches.
                pair_pos_items.append(self._encode_one(messages, pos_ans, answer_loss_mode="answer_only" if pos_ans else "full"))
                neg_w.append(w)
                neg_source.append(source_label_id)
                neg_neg_type.append(neg_type_id)
                if self.track_fourway_category:
                    neg_cat.append(NEG_FOURWAY_IDS.get(category, -1))

        batch: Dict[str, torch.Tensor] = {}
        if pos_items:
            pi, pm, pl, pa, pt = self._pad_stack(pos_items)
            batch["pos_input_ids"] = pi
            batch["pos_attention_mask"] = pm
            batch["pos_labels"] = pl
            batch["pos_weights"] = torch.tensor(pos_w, dtype=torch.float)
            batch["pos_answer_mask"] = pa
            batch["pos_template_mask"] = pt
            batch["pos_source_labels"] = torch.tensor(pos_source, dtype=torch.long)
            batch["pos_neg_type_ids"] = torch.tensor(pos_neg_type, dtype=torch.long)
            if self.track_fourway_category:
                batch["pos_category_ids"] = torch.tensor(pos_cat, dtype=torch.long)
        if neg_items:
            ni, nm, nl, na, nt = self._pad_stack(neg_items)
            batch["neg_input_ids"] = ni
            batch["neg_attention_mask"] = nm
            batch["neg_labels"] = nl
            batch["neg_weights"] = torch.tensor(neg_w, dtype=torch.float)
            batch["neg_answer_mask"] = na
            batch["neg_template_mask"] = nt
            batch["neg_source_labels"] = torch.tensor(neg_source, dtype=torch.long)
            batch["neg_neg_type_ids"] = torch.tensor(neg_neg_type, dtype=torch.long)
            if pair_pos_items:
                ppi, ppm, ppl, ppa, ppt = self._pad_stack(pair_pos_items)
                batch["pair_pos_input_ids"] = ppi
                batch["pair_pos_attention_mask"] = ppm
                batch["pair_pos_labels"] = ppl
                batch["pair_pos_answer_mask"] = ppa
                batch["pair_source_labels"] = torch.tensor(neg_source, dtype=torch.long)
                batch["pair_neg_type_ids"] = torch.tensor(neg_neg_type, dtype=torch.long)
            if self.track_fourway_category:
                batch["neg_category_ids"] = torch.tensor(neg_cat, dtype=torch.long)

        if not batch:
            batch["pos_input_ids"] = torch.zeros((1, 1), dtype=torch.long)
            batch["pos_attention_mask"] = torch.ones((1, 1), dtype=torch.long)
            batch["pos_labels"] = torch.full((1, 1), -100, dtype=torch.long)
            batch["pos_weights"] = torch.ones((1,), dtype=torch.float)
            batch["pos_answer_mask"] = torch.zeros((1, 1), dtype=torch.long)
            batch["pos_template_mask"] = torch.zeros((1, 1), dtype=torch.long)
            batch["pos_source_labels"] = torch.full((1,), -1, dtype=torch.long)
            batch["pos_neg_type_ids"] = torch.zeros((1,), dtype=torch.long)
        return batch


# -------------------------
# Trainer
# -------------------------
class DebateULTrainer(Trainer):
    def __init__(
        self,
        *args,
        alpha_ul: float = 0.8,
        category_balancing: str = "none",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        c = TRAIN_METHOD_CONSTANTS
        self.alpha_ul = float(alpha_ul)
        self.ul_eps = float(c["ul_eps"])
        self.category_balancing = str(category_balancing or "none")
        self.pos_answer_ce_weight = float(c["pos_answer_ce_weight"])
        self.pos_template_ce_weight = float(c["pos_template_ce_weight"])
        self.pos_late_token_boost = float(c["pos_late_token_boost"])
        self.pos_early_terminate_weight = float(c["pos_early_terminate_weight"])
        self.source_cls_weight = float(c["source_cls_weight"])
        self.internal_pos_ce_bonus = float(c["internal_pos_ce_bonus"])
        self.internal_answer_ce_bonus = float(c["internal_answer_ce_bonus"])
        self.internal_neg_ul_bonus = float(c["internal_neg_ul_bonus"])
        self.internal_idk_margin_weight = float(c["internal_idk_margin_weight"])
        self.internal_idk_margin = float(c["internal_idk_margin"])
        self.internal_meta_margin_weight = float(c["internal_meta_margin_weight"])
        self.internal_meta_margin = float(c["internal_meta_margin"])
        self.internal_final_margin_weight = float(c["internal_final_margin_weight"])
        self.internal_final_margin = float(c["internal_final_margin"])
        self.source_cls_internal_weight = float(c["source_cls_internal_weight"])
        self.source_cls_external_weight = float(c["source_cls_external_weight"])
        self.source_cls_both_weight = float(c["source_cls_both_weight"])
        self.source_cls_abstain_weight = float(c["source_cls_abstain_weight"])
        self.source_cls_ignore_abstain = bool(c["source_cls_ignore_abstain"])
        self.enable_internal_ema_beta = bool(c["enable_internal_ema_beta"])
        self.internal_ema_alpha = float(c["internal_ema_alpha"])
        self.internal_ema_beta_min = float(c["internal_ema_beta_min"])

        self.internal_ema_baseline: Optional[float] = None
        self.last_internal_ema_baseline: float = -1.0
        self.last_internal_ema_scale: float = 1.0
        self.terminator_token_ids = self._build_terminator_token_ids()

    @staticmethod
    def _token_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        vocab = shift_logits.size(-1)
        loss_flat = F.cross_entropy(
            shift_logits.view(-1, vocab),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        )
        loss_tok = loss_flat.view(shift_labels.size())
        mask = (shift_labels != -100).float()
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (loss_tok * mask).sum(dim=1) / denom

    def _token_unlikelihood_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        mask = shift_labels != -100
        probs = F.softmax(shift_logits, dim=-1)
        tgt = shift_labels.clamp_min(0).unsqueeze(-1)
        p = probs.gather(-1, tgt).squeeze(-1).clamp(self.ul_eps, 1.0 - self.ul_eps)
        ul = -torch.log(1.0 - p)
        mask_f = mask.float()
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return (ul * mask_f).sum(dim=1) / denom

    def _build_terminator_token_ids(self) -> List[int]:
        ids: List[int] = []
        tok = getattr(self, "tokenizer", None)
        if tok is None:
            return ids
        eos_id = getattr(tok, "eos_token_id", None)
        if eos_id is not None:
            ids.append(int(eos_id))
        try:
            close_ids = tok("</answer>", add_special_tokens=False, return_attention_mask=False)["input_ids"]
            if isinstance(close_ids, list) and close_ids:
                ids.append(int(close_ids[0]))
        except Exception:
            pass
        # keep unique stable order
        out: List[int] = []
        seen = set()
        for x in ids:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out


    def _paired_margin_loss(
        self,
        pos_ce: torch.Tensor,
        neg_ce: torch.Tensor,
        source_labels: Optional[torch.Tensor],
        neg_type_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if source_labels is None or neg_type_ids is None or pos_ce.numel() == 0 or neg_ce.numel() == 0:
            return pos_ce.new_tensor(0.0)
        n = min(pos_ce.size(0), neg_ce.size(0), source_labels.size(0), neg_type_ids.size(0))
        if n <= 0:
            return pos_ce.new_tensor(0.0)
        pos_score = -pos_ce[:n]
        neg_score = -neg_ce[:n]
        src = source_labels[:n]
        neg_t = neg_type_ids[:n]
        total = pos_ce.new_tensor(0.0)

        idk_mask = (src == 0) & (neg_t == 1)
        if idk_mask.any() and self.internal_idk_margin_weight > 0.0:
            gap = pos_score[idk_mask] - neg_score[idk_mask]
            total = total + self.internal_idk_margin_weight * F.relu(self.internal_idk_margin - gap).mean()

        meta_mask = (src == 0) & (neg_t == 4)
        if meta_mask.any() and self.internal_meta_margin_weight > 0.0:
            gap = pos_score[meta_mask] - neg_score[meta_mask]
            total = total + self.internal_meta_margin_weight * F.relu(self.internal_meta_margin - gap).mean()

        # v21: final-like hard negatives include wrong final answers, wrong candidates,
        # wrong-context answers, and debate-drift answers. Apply this only to internal
        # fallback/source=internal pairs to avoid globally suppressing context usage.
        final_like_mask = (src == 0) & ((neg_t == 2) | (neg_t == 5) | (neg_t == 6) | (neg_t == 7) | (neg_t == 8))
        if final_like_mask.any() and self.internal_final_margin_weight > 0.0:
            gap = pos_score[final_like_mask] - neg_score[final_like_mask]
            total = total + self.internal_final_margin_weight * F.relu(self.internal_final_margin - gap).mean()

        return total


    def _answer_core_ce_loss(self, logits: torch.Tensor, labels: torch.Tensor, answer_mask: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_answer = (answer_mask[:, 1:].contiguous() > 0) & (shift_labels != -100)

        vocab = shift_logits.size(-1)
        loss_flat = F.cross_entropy(
            shift_logits.view(-1, vocab),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        )
        loss_tok = loss_flat.view(shift_labels.size())

        mask = shift_answer.float()
        if self.pos_late_token_boost > 1.0:
            pos_index = torch.cumsum(mask, dim=1)
            answer_len = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            denom = (answer_len - 1.0).clamp_min(1.0)
            frac = ((pos_index - 1.0) / denom).clamp_min(0.0)
            token_w = (1.0 + (self.pos_late_token_boost - 1.0) * frac) * mask
        else:
            token_w = mask

        denom = token_w.sum(dim=1).clamp_min(1.0)
        return (loss_tok * token_w).sum(dim=1) / denom

    def _template_ce_loss(self, logits: torch.Tensor, labels: torch.Tensor, template_mask: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_template = (template_mask[:, 1:].contiguous() > 0) & (shift_labels != -100)
        if shift_template.sum() <= 0:
            return torch.zeros((logits.size(0),), device=logits.device, dtype=logits.dtype)

        vocab = shift_logits.size(-1)
        loss_flat = F.cross_entropy(
            shift_logits.view(-1, vocab),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        )
        loss_tok = loss_flat.view(shift_labels.size())
        mask = shift_template.float()
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (loss_tok * mask).sum(dim=1) / denom

    def _scale_weights_by_source(self, weights: torch.Tensor, source_labels: Optional[torch.Tensor], multipliers: Dict[int, float]) -> torch.Tensor:
        if source_labels is None:
            return weights
        out = weights.clone()
        for sid, mult in multipliers.items():
            if mult == 1.0:
                continue
            mask = source_labels == int(sid)
            if mask.any():
                out = torch.where(mask, out * float(mult), out)
        return out

    def _get_internal_beta_floor(self) -> float:
        """Fixed EMA lower-bound scale for internal-conflict examples."""
        return float(self.internal_ema_beta_min)

    def _internal_ema_scales(
        self,
        current_losses: torch.Tensor,
        source_labels: Optional[torch.Tensor],
        neg_type_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        scales = torch.ones_like(current_losses)
        if (not self.enable_internal_ema_beta) or source_labels is None or neg_type_ids is None or current_losses.numel() == 0:
            self.last_internal_ema_scale = 1.0
            return scales
        n = min(current_losses.size(0), source_labels.size(0), neg_type_ids.size(0))
        if n <= 0:
            self.last_internal_ema_scale = 1.0
            return scales
        losses = current_losses[:n].detach()
        src = source_labels[:n]
        neg_t = neg_type_ids[:n]
        mask = (src == 0) & (neg_t != 0)
        if mask.sum() <= 0:
            self.last_internal_ema_scale = 1.0
            return scales
        selected = losses[mask]
        current_mean = float(selected.mean().item())
        if self.internal_ema_baseline is None:
            self.internal_ema_baseline = current_mean
        baseline = float(self.internal_ema_baseline)
        self.last_internal_ema_baseline = baseline

        beta_floor = self._get_internal_beta_floor()

        local_scales = torch.ones_like(selected)
        worse = selected > baseline
        if worse.any():
            ratio = baseline / selected[worse].clamp_min(1e-8)
            ratio = torch.clamp(ratio, min=beta_floor, max=1.0)
            local_scales[worse] = ratio.to(local_scales.dtype)
        scales[:n][mask] = local_scales
        self.last_internal_ema_scale = float(local_scales.mean().item())
        self.internal_ema_baseline = (1.0 - self.internal_ema_alpha) * baseline + self.internal_ema_alpha * current_mean
        return scales

    def _source_cls_loss(self, source_logits: torch.Tensor, source_labels: torch.Tensor) -> torch.Tensor:
        valid = source_labels >= 0
        if self.source_cls_ignore_abstain:
            valid = valid & (source_labels != 3)
        if valid.sum() <= 0:
            return torch.zeros((), device=source_logits.device, dtype=source_logits.dtype)

        ce = F.cross_entropy(source_logits, source_labels.clamp_min(0), reduction="none")
        class_w = torch.ones_like(ce)
        class_w = torch.where(source_labels == 0, class_w * self.source_cls_internal_weight, class_w)
        class_w = torch.where(source_labels == 1, class_w * self.source_cls_external_weight, class_w)
        class_w = torch.where(source_labels == 2, class_w * self.source_cls_both_weight, class_w)
        class_w = torch.where(source_labels == 3, class_w * self.source_cls_abstain_weight, class_w)
        ce = ce * class_w * valid.float()
        denom = (class_w * valid.float()).sum().clamp_min(1.0)
        return ce.sum() / denom

    def _early_terminate_loss(self, logits: torch.Tensor, labels: torch.Tensor, answer_mask: torch.Tensor) -> torch.Tensor:
        if not self.terminator_token_ids:
            return torch.zeros((logits.size(0),), device=logits.device, dtype=logits.dtype)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_answer = (answer_mask[:, 1:].contiguous() > 0) & (shift_labels != -100)
        mask = shift_answer.float()
        if mask.sum() <= 0:
            return torch.zeros((logits.size(0),), device=logits.device, dtype=logits.dtype)

        probs = F.softmax(shift_logits, dim=-1)
        term_p = probs[..., self.terminator_token_ids].sum(dim=-1).clamp(self.ul_eps, 1.0 - self.ul_eps)
        loss_tok = -torch.log(1.0 - term_p)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (loss_tok * mask).sum(dim=1) / denom

    @staticmethod
    def _aggregate_group_losses(per_sample: torch.Tensor, weights: torch.Tensor, category_ids: Optional[torch.Tensor]) -> torch.Tensor:
        if category_ids is None:
            return (per_sample * weights).mean()
        present_losses = []
        valid_ids = torch.unique(category_ids)
        for cid in valid_ids.tolist():
            if cid < 0:
                continue
            mask = category_ids == cid
            if mask.any():
                present_losses.append((per_sample[mask] * weights[mask]).mean())
        if not present_losses:
            return (per_sample * weights).mean()
        return torch.stack(present_losses).mean()

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        logs = dict(logs)
        if self.enable_internal_ema_beta:
            logs.setdefault("internal_ema_scale", round(float(self.last_internal_ema_scale), 6))
            logs.setdefault("internal_ema_baseline", round(float(self.last_internal_ema_baseline), 6))
        return super().log(logs, *args, **kwargs)

    def compute_loss(self, model, inputs: Dict[str, torch.Tensor], return_outputs: bool = False, num_items_in_batch: Optional[torch.Tensor] = None, **kwargs):
        device = next(model.parameters()).device
        total_loss = torch.tensor(0.0, device=device)
        outputs_out = None
        pos_answer_for_pair: Optional[torch.Tensor] = None
        pos_source_for_pair: Optional[torch.Tensor] = None
        pos_neg_type_for_pair: Optional[torch.Tensor] = None

        if "pos_input_ids" in inputs:
            pos_ids = inputs["pos_input_ids"].to(device)
            pos_mask = inputs["pos_attention_mask"].to(device)
            pos_labels = inputs["pos_labels"].to(device)
            pos_answer_mask = inputs.get("pos_answer_mask")
            pos_answer_mask = pos_answer_mask.to(device) if pos_answer_mask is not None else torch.zeros_like(pos_labels)
            pos_template_mask = inputs.get("pos_template_mask")
            pos_template_mask = pos_template_mask.to(device) if pos_template_mask is not None else torch.zeros_like(pos_labels)
            pos_source_labels = inputs.get("pos_source_labels")
            pos_source_labels = pos_source_labels.to(device) if pos_source_labels is not None else None
            pos_neg_type_ids = inputs.get("pos_neg_type_ids")
            pos_neg_type_ids = pos_neg_type_ids.to(device) if pos_neg_type_ids is not None else None
            pos_w = inputs.get("pos_weights", torch.ones((pos_ids.size(0),), device=device)).to(device)
            pos_cat = inputs.get("pos_category_ids")
            pos_cat = pos_cat.to(device) if pos_cat is not None else None

            pos_w_main = self._scale_weights_by_source(
                pos_w,
                pos_source_labels,
                {0: self.internal_pos_ce_bonus},
            )
            pos_w_answer = self._scale_weights_by_source(
                pos_w_main,
                pos_source_labels,
                {0: self.internal_answer_ce_bonus},
            )

            out = model(input_ids=pos_ids, attention_mask=pos_mask)
            logits = out.logits
            per = self._token_ce_loss(logits, pos_labels)
            per_answer_for_ema = self._answer_core_ce_loss(logits, pos_labels, pos_answer_mask)
            pos_answer_for_pair = per_answer_for_ema
            pos_source_for_pair = pos_source_labels
            pos_neg_type_for_pair = pos_neg_type_ids
            ema_scales = self._internal_ema_scales(per_answer_for_ema, pos_source_labels, pos_neg_type_ids)
            pos_w_main = pos_w_main * ema_scales
            pos_w_answer = pos_w_answer * ema_scales
            pos_w_term = pos_w * ema_scales
            pos_loss = self._aggregate_group_losses(per, pos_w_main, pos_cat if self.category_balancing == "fourway_equal" else None)
            total_loss = total_loss + pos_loss

            if self.pos_template_ce_weight > 0.0:
                per_template = self._template_ce_loss(logits, pos_labels, pos_template_mask)
                pos_template_loss = self._aggregate_group_losses(per_template, pos_w_main, pos_cat if self.category_balancing == "fourway_equal" else None)
                total_loss = total_loss + (self.pos_template_ce_weight * pos_template_loss)

            if self.pos_answer_ce_weight > 0.0:
                per_answer = per_answer_for_ema
                pos_answer_loss = self._aggregate_group_losses(per_answer, pos_w_answer, pos_cat if self.category_balancing == "fourway_equal" else None)
                total_loss = total_loss + (self.pos_answer_ce_weight * pos_answer_loss)

            if self.pos_early_terminate_weight > 0.0:
                per_term = self._early_terminate_loss(logits, pos_labels, pos_answer_mask)
                pos_term_loss = self._aggregate_group_losses(per_term, pos_w_term, pos_cat if self.category_balancing == "fourway_equal" else None)
                total_loss = total_loss + (self.pos_early_terminate_weight * pos_term_loss)

            if self.source_cls_weight > 0.0 and pos_source_labels is not None and hasattr(out, "source_logits"):
                src_loss = self._source_cls_loss(out.source_logits, pos_source_labels)
                total_loss = total_loss + (self.source_cls_weight * src_loss)

            outputs_out = out

        if "neg_input_ids" in inputs:
            neg_ids = inputs["neg_input_ids"].to(device)
            neg_mask = inputs["neg_attention_mask"].to(device)
            neg_labels = inputs["neg_labels"].to(device)
            neg_answer_mask = inputs.get("neg_answer_mask")
            neg_answer_mask = neg_answer_mask.to(device) if neg_answer_mask is not None else torch.zeros_like(neg_labels)
            neg_source_labels = inputs.get("neg_source_labels")
            neg_source_labels = neg_source_labels.to(device) if neg_source_labels is not None else None
            neg_w = inputs.get("neg_weights", torch.ones((neg_ids.size(0),), device=device)).to(device)
            neg_w = self._scale_weights_by_source(
                neg_w,
                neg_source_labels,
                {0: self.internal_neg_ul_bonus},
            )
            neg_neg_type_ids = inputs.get("neg_neg_type_ids")
            neg_neg_type_ids = neg_neg_type_ids.to(device) if neg_neg_type_ids is not None else None
            neg_cat = inputs.get("neg_category_ids")
            neg_cat = neg_cat.to(device) if neg_cat is not None else None

            out2 = model(input_ids=neg_ids, attention_mask=neg_mask)
            logits2 = out2.logits
            per_ul = self._token_unlikelihood_loss(logits2, neg_labels)
            neg_loss = self._aggregate_group_losses(per_ul, neg_w, neg_cat if self.category_balancing == "fourway_equal" else None)
            total_loss = total_loss + (self.alpha_ul * neg_loss)

            # v21: pairwise answer-core margin for internal fallback / anti-drift pairs.
            # Use aligned pair_pos_* tensors from the collator, rather than the mixed
            # positive batch, to avoid pairing a negative with an unrelated clean-CE record.
            if "pair_pos_input_ids" in inputs and self.internal_final_margin_weight > 0.0:
                pair_ids = inputs["pair_pos_input_ids"].to(device)
                pair_mask = inputs["pair_pos_attention_mask"].to(device)
                pair_labels = inputs["pair_pos_labels"].to(device)
                pair_answer_mask = inputs["pair_pos_answer_mask"].to(device)
                pair_source = inputs.get("pair_source_labels")
                pair_source = pair_source.to(device) if pair_source is not None else neg_source_labels
                pair_neg_type = inputs.get("pair_neg_type_ids")
                pair_neg_type = pair_neg_type.to(device) if pair_neg_type is not None else neg_neg_type_ids
                if pair_neg_type is not None:
                    pair_neg_type = pair_neg_type.to(device)
                pair_out = model(input_ids=pair_ids, attention_mask=pair_mask)
                pair_pos_ce = self._answer_core_ce_loss(pair_out.logits, pair_labels, pair_answer_mask)
                neg_answer_ce = self._answer_core_ce_loss(logits2, neg_labels, neg_answer_mask)
                margin_loss = self._paired_margin_loss(pair_pos_ce, neg_answer_ce, pair_source, pair_neg_type)
                total_loss = total_loss + margin_loss

        return (total_loss, outputs_out) if return_outputs else total_loss


# -------------------------
# CLI
# -------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--load_in_4bit", action="store_true")

    p.add_argument(
        "--objective",
        type=str,
        default="auto",
        choices=["auto", "sft", "balanced_ul", "debate_4way"],
        help="auto will pick debate_4way/balanced_ul when dataset has pos_answer or neg_answer.",
    )
    p.add_argument("--alpha_ul", type=float, default=0.8)
    p.add_argument(
        "--category_balancing",
        type=str,
        default="none",
        choices=["none", "fourway_equal"],
        help="For debate_4way, use fourway_equal to average present class losses equally within a batch.",
    )

    p.add_argument(
        "--device_map",
        type=str,
        default="cuda",
        help="'cuda' uses LOCAL_RANK in torchrun/DDP; 'auto' enables HF auto device_map for single-process offload.",
    )
    p.add_argument("--offload_folder", type=str, default=None)
    p.add_argument("--offload_buffers", action="store_true")

    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--val_split", type=float, default=0.0)
    p.add_argument("--system_prompt", type=str, default=None)
    p.add_argument("--system_prompt_file", type=str, default=None)
    p.add_argument("--use_context_label", action="store_true")
    p.add_argument("--max_samples", type=int, default=None)

    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    p.add_argument("--use_rslora", action="store_true")

    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=-1, help="Keep -1 so num_train_epochs is respected. Positive values override epoch control in HF Trainer.")
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--dataset_num_proc", type=int, default=2)
    p.add_argument("--dataloader_num_workers", type=int, default=2)
    p.add_argument("--packing", action="store_true")
    p.add_argument("--save_merged", action="store_true")
    p.add_argument("--init_adapter_path", type=str, default=None, help="Optional LoRA adapter path to initialize / continue training from a previous stage.")
    return p.parse_args(argv)


# -------------------------
# Dataset loaders
# -------------------------
def _peek_first_record(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    return json.loads(line)
        return {}
    obj = load_records(path)
    return obj[0] if obj else {}


def _load_ul_dataset(path: str, max_samples: Optional[int] = None) -> Dataset:
    records = load_records(path)
    if max_samples is not None:
        records = records[: int(max_samples)]

    kept = []
    for r in records:
        if not isinstance(r, dict):
            continue
        if "pos_answer" not in r and "neg_answer" not in r and ("stage1" in r and "stage2" in r):
            cat = _infer_category_from_stage_fields(r)
            if cat is None or r.get("excluded_from_stats"):
                continue
            stage2 = r["stage2"]
            critic = stage2.get("critic", {})
            has_rebuttal = critic.get("has_rebuttal") == "Yes"
            final_out = (stage2.get("revision_raw") if has_rebuttal else stage2.get("llm1_raw") or "").strip()
            sys_prompt = r["stage1"]["prompts"]["llm1_meta"]
            user_prompt = stage2["prompts"].get("llm1_first") or ""
            r = {
                "category": cat,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "pos_answer": final_out if cat in ("S11", "S21") else "",
                "neg_answer": final_out if cat in ("S12", "S22") else "",
                "sample_weight": float(r.get("sample_weight", 1.0)),
            }
        kept.append(r)
    return Dataset.from_list(kept)


def _infer_objective(first: Dict[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    has_ul_fields = ("pos_answer" in first) or ("neg_answer" in first) or ("stage1" in first and "stage2" in first)
    if not has_ul_fields:
        return "sft"
    cat = str(first.get("category", ""))
    if cat.startswith("external_"):
        return "debate_4way"
    return "balanced_ul"


# -------------------------
# Main
# -------------------------
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank if world_size > 1 else 0)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if _is_bfloat16_supported() else torch.float16,
        )

    model_kwargs = dict(
        trust_remote_code=True,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if _is_bfloat16_supported() else torch.float16,
    )
    dm = (args.device_map or "").lower().strip()
    if dm in ("cuda", "gpu", "cuda:0", "0", "ddp"):
        model_kwargs["device_map"] = {"": local_rank if world_size > 1 else 0}
    elif dm in ("auto", "balanced", "sequential"):
        if world_size > 1:
            model_kwargs["device_map"] = {"": local_rank}
        else:
            model_kwargs["device_map"] = "auto"
    elif dm in ("none", "null"):
        model_kwargs["device_map"] = None
    elif dm:
        model_kwargs["device_map"] = args.device_map

    # More stable than xformers/flash for the user's repeated two-stage issues.
    model_kwargs["attn_implementation"] = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    model.config.use_cache = False

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()

    if args.init_adapter_path:
        model = PeftModel.from_pretrained(model, args.init_adapter_path, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=target_modules,
            task_type="CAUSAL_LM",
            use_rslora=bool(args.use_rslora),
        )
        model = get_peft_model(model, peft_config)

    if TRAIN_METHOD_CONSTANTS["source_cls_weight"] > 0.0:
        model = SourceAwareModel(model, num_labels=4, dropout=TRAIN_METHOD_CONSTANTS["source_cls_dropout"])

    first = _peek_first_record(args.dataset_path)
    objective = _infer_objective(first, args.objective)
    category_balancing = args.category_balancing
    if objective == "debate_4way" and category_balancing == "none":
        category_balancing = "fourway_equal"

    # Shared training args for trainer-based modes
    def _common_training_args(eval_mode: bool = False) -> TrainingArguments:
        return TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            fp16=not _is_bfloat16_supported(),
            bf16=_is_bfloat16_supported(),
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_strategy="steps" if eval_mode else "no",
            eval_steps=args.save_steps if eval_mode else None,
            optim="adamw_torch",
            weight_decay=args.weight_decay,
            lr_scheduler_type="linear",
            seed=args.seed,
            report_to="none",
            remove_unused_columns=False,
            ddp_find_unused_parameters=False if world_size > 1 else None,
            dataloader_num_workers=args.dataloader_num_workers,
            save_on_each_node=False,
            gradient_checkpointing=False,
        )

    if objective in {"balanced_ul", "debate_4way"}:
        dataset = _load_ul_dataset(args.dataset_path, max_samples=args.max_samples)
        collator = DebateULDataCollator(
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            track_fourway_category=(objective == "debate_4way"),
            ignore_reason_loss=bool(TRAIN_METHOD_CONSTANTS["ignore_reason_loss"]),
            neg_loss_on_answer_only=bool(TRAIN_METHOD_CONSTANTS["neg_loss_on_answer_only"]),
        )
        trainer = DebateULTrainer(
            model=model,
            args=_common_training_args(eval_mode=False),
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=collator,
            alpha_ul=args.alpha_ul,
            category_balancing=category_balancing,
        )
        trainer.train()
    else:
        system_prompt = args.system_prompt
        if args.system_prompt_file:
            system_prompt = _read_text_file(args.system_prompt_file)

        custom_dataset = LocalJsonDataset(
            json_file=args.dataset_path,
            processing_class=tokenizer,
            max_seq_length=args.max_seq_length,
            system_prompt=system_prompt,
            use_context_label=bool(args.use_context_label),
            max_samples=args.max_samples,
        )
        dataset = custom_dataset.get_dataset()
        if args.val_split and 0.0 < args.val_split < 0.5:
            split = dataset.train_test_split(test_size=args.val_split, seed=args.seed)
            train_dataset = split["train"]
            eval_dataset = split["test"]
        else:
            train_dataset = dataset
            eval_dataset = None

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            dataset_num_proc=args.dataset_num_proc,
            packing=bool(args.packing),
            args=_common_training_args(eval_mode=(eval_dataset is not None)),
        )
        trainer.train()

    if trainer.is_world_process_zero():
        adapter_dir = os.path.join(args.output_dir, "adapter")
        save_model = model.base_model if isinstance(model, SourceAwareModel) else model
        save_model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        if args.save_merged:
            merged_dir = os.path.join(args.output_dir, "merged")
            os.makedirs(merged_dir, exist_ok=True)
            try:
                merged = save_model.merge_and_unload()
                merged.save_pretrained(merged_dir, safe_serialization=True)
                tokenizer.save_pretrained(merged_dir)
            except Exception as e:
                print(f"[WARN] merge_and_unload failed: {e}")


if __name__ == "__main__":
    main()
