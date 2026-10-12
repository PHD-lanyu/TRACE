"""QA debate utilities for ConFiQA-style experiments.

This module supports:
- question-only answering with LLM1
- multi-round debate (LLM1 responder, LLM2 critic, LLM3 supervisor/judge)
- answer normalization / exact-ish string matching for QA tasks
- detailed debate trace export for later retrieval-material construction
"""

from __future__ import annotations

import ast
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .agent import Agent


# ------------------------- Generic parsing helpers -------------------------


def parse_structured_dict(text: str) -> dict:
    """Parse a dict/JSON object from an LLM response."""
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text)}")

    s = text.strip()
    if s.startswith("```"):
        m = re.search(r"```(?:json|python|text)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
        if m:
            s = m.group(1).strip()

    if "{" in s and "}" in s:
        s2 = s[s.find("{") : s.rfind("}") + 1].strip()
    else:
        s2 = s

    try:
        obj = json.loads(s2)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    try:
        obj = ast.literal_eval(s2)
        if isinstance(obj, dict):
            return obj
    except Exception as e:
        raise ValueError(f"Failed to parse dict from model output. First 200 chars: {text[:200]!r}") from e

    raise ValueError(f"Parsed object is not a dict. First 200 chars: {text[:200]!r}")


_ARTICLE_RE = re.compile(r"\bthe\b", flags=re.IGNORECASE)


def normalize_answer_text(text: str) -> str:
    s = str(text or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = _ARTICLE_RE.sub("", s)
    s = s.replace("_", " ")
    s = s.replace(".", " ")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", s)
    return s


def dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        s = str(x or "").strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def collect_gold_answers(example: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []

    answer = example.get("answer")
    if isinstance(answer, list):
        candidates.extend(str(x) for x in answer if str(x).strip())
    elif isinstance(answer, str) and answer.strip():
        candidates.append(answer)

    for key in ("orig_answer",):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value)

    for key in ("orig_alias",):
        value = example.get(key)
        if isinstance(value, list):
            candidates.extend(str(x) for x in value if str(x).strip())

    return dedupe_preserve_order(candidates)


def answer_is_correct(prediction: str, gold_answers: Sequence[str]) -> Tuple[bool, Optional[str], str, List[str]]:
    pred_norm = normalize_answer_text(prediction)
    gold_list = dedupe_preserve_order([str(x) for x in gold_answers if str(x).strip()])
    gold_norms = [normalize_answer_text(x) for x in gold_list]

    for raw_gold, norm_gold in zip(gold_list, gold_norms):
        if pred_norm and pred_norm == norm_gold:
            return True, raw_gold, pred_norm, gold_norms
    return False, None, pred_norm, gold_norms


# ------------------------- QA output parsing -------------------------


def _extract_answer_from_text_fallback(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""

    for line in raw.splitlines():
        line = line.strip().strip("`")
        if not line:
            continue
        lowered = line.lower()
        for prefix in (
            "short_answer:",
            "short answer:",
            "final_answer:",
            "final answer:",
            "answer:",
        ):
            if lowered.startswith(prefix):
                return line.split(":", 1)[1].strip()

    first = next((ln.strip().strip("`") for ln in raw.splitlines() if ln.strip()), raw)
    first = re.sub(r"^(short\s+answer|final\s+answer|answer)\s*[:\-]\s*", "", first, flags=re.IGNORECASE)
    return first.strip()


def parse_answer_payload(text: str) -> Tuple[Dict[str, Any], str, bool, str]:
    raw = str(text or "")
    try:
        obj = parse_structured_dict(raw)
        for key in ("short_answer", "final_answer", "answer"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return obj, value.strip(), False, f"json_{key}"
        fallback = _extract_answer_from_text_fallback(raw)
        return obj, fallback, True, "json_missing_answer_field"
    except Exception:
        fallback = _extract_answer_from_text_fallback(raw)
        if fallback:
            return {}, fallback, True, "text_fallback"
        return {}, "", True, "failed_to_extract_answer"


def parse_critic_payload(text: str) -> Tuple[Dict[str, Any], bool, str]:
    try:
        return parse_structured_dict(text), False, ""
    except Exception as e:
        return {}, True, str(e)


def parse_judge_payload(text: str) -> Tuple[Dict[str, Any], bool, str]:
    try:
        return parse_structured_dict(text), False, ""
    except Exception as e:
        return {}, True, str(e)


# ------------------------- Debate config -------------------------


@dataclass
class RunConfig:
    responder_model: str
    critic_model: str
    judge_model: str
    temperature: float
    api_key: str
    sleep_time: float
    max_rounds: int = 5
    responder_device: Optional[str] = None
    responder_base_url: Optional[str] = None
    critic_base_url: Optional[str] = None
    judge_base_url: Optional[str] = None


def render_template(tpl: str, **kwargs: str) -> str:
    s = str(tpl)
    for k, v in kwargs.items():
        s = s.replace(f"##{k}##", str(v))
    return s


def _make_agent(
    *,
    model_name: str,
    name: str,
    temperature: float,
    sleep_time: float,
    api_key: str,
    base_url: Optional[str] = None,
    hf_device: Optional[str] = None,
) -> Agent:
    agent = Agent(model_name=model_name, name=name, temperature=temperature, sleep_time=sleep_time)
    agent.openai_api_key = api_key
    if base_url:
        agent.ollama_base_url = base_url
    if hf_device:
        agent.hf_device = hf_device
    return agent


def _candidate_for_output(
    *,
    round_id: int,
    raw_text: str,
    payload: Dict[str, Any],
    short_answer: str,
    parse_error: bool,
    parse_reason: str,
) -> Dict[str, Any]:
    return {
        "round_id": round_id,
        "raw": raw_text,
        "payload": payload,
        "short_answer": short_answer,
        "normalized_short_answer": normalize_answer_text(short_answer),
        "parse_error": parse_error,
        "parse_reason": parse_reason,
    }


def _debate_history_text(rounds: Sequence[Dict[str, Any]]) -> str:
    if not rounds:
        return "No previous rounds."
    lines: List[str] = []
    for rd in rounds:
        cand = rd.get("candidate") or {}
        critic = rd.get("critic") or {}
        judge_review = rd.get("judge_review") or {}
        lines.append(f"Round {cand.get('round_id')}: candidate_answer={cand.get('short_answer', '')}")
        if isinstance(critic, dict) and critic:
            lines.append(
                f"Round {cand.get('round_id')} critic: has_rebuttal={critic.get('has_rebuttal', '')}; "
                f"recommended_action={critic.get('recommended_action', '')}; "
                f"suggested_answer={critic.get('suggested_answer', '')}"
            )
            issues = critic.get("issues")
            if isinstance(issues, list) and issues:
                lines.append("Key critic issues: " + " | ".join(str(x) for x in issues[:4]))
        if isinstance(judge_review, dict) and judge_review:
            lines.append(
                f"Round {cand.get('round_id')} judge_review: should_stop={judge_review.get('should_stop', '')}; "
                f"current_best_answer={judge_review.get('current_best_answer', '')}; "
                f"reason={judge_review.get('reason', '')}"
            )
            revision_focus = judge_review.get("revision_focus")
            if isinstance(revision_focus, str) and revision_focus.strip():
                lines.append("Judge revision focus: " + revision_focus.strip())
    return "\n".join(lines)


def _candidate_catalog_text(candidates: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for cand in candidates:
        lines.append(
            json.dumps(
                {
                    "round_id": cand.get("round_id"),
                    "short_answer": cand.get("short_answer"),
                    "payload": cand.get("payload", {}),
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


def _judge_review_requests_stop(judge_payload: Dict[str, Any], judge_parse_error: bool) -> Tuple[bool, str]:
    if judge_parse_error:
        return False, "judge_review_parse_error_continue"

    should_stop = str(judge_payload.get("should_stop", "")).strip().lower()
    if should_stop in {"yes", "true", "stop", "terminate"}:
        return True, "judge_review_stop"
    return False, "judge_review_continue"


def _finalize_from_judge_payload(
    judge_payload: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
) -> Tuple[Optional[int], str, str, str]:
    if not candidates:
        return None, "", "no_candidates", ""

    candidate_by_id = {int(c.get("round_id", 0)): c for c in candidates if int(c.get("round_id", 0)) > 0}

    selected_id_raw = judge_payload.get("selected_candidate_id")
    selected_id: Optional[int] = None
    if isinstance(selected_id_raw, int):
        selected_id = selected_id_raw
    elif isinstance(selected_id_raw, str):
        m = re.search(r"(\d+)", selected_id_raw)
        if m:
            selected_id = int(m.group(1))

    summary = str(judge_payload.get("summary") or judge_payload.get("debate_summary") or "").strip()
    final_answer = str(judge_payload.get("final_answer") or "").strip()

    if selected_id in candidate_by_id:
        chosen = candidate_by_id[selected_id]
        if not final_answer:
            final_answer = str(chosen.get("short_answer") or "").strip()
        return selected_id, final_answer, "judge_selected_candidate", summary

    if final_answer:
        return None, final_answer, "judge_final_answer_only", summary

    last = candidates[-1]
    return int(last.get("round_id", 0)), str(last.get("short_answer") or "").strip(), "fallback_last_candidate", summary


# ------------------------- Stage runners -------------------------


def run_responder_only_qa(
    *,
    question: str,
    retrieved_context: str,
    prompt_cfg: Dict[str, Any],
    run_cfg: RunConfig,
    include_prompts: bool = False,
) -> Dict[str, Any]:
    ctx = (retrieved_context or "").strip()

    llm1 = _make_agent(
        model_name=run_cfg.responder_model,
        name="LLM1",
        temperature=run_cfg.temperature,
        sleep_time=run_cfg.sleep_time,
        api_key=run_cfg.api_key,
        base_url=run_cfg.responder_base_url,
        hf_device=run_cfg.responder_device,
    )
    llm1.set_meta_prompt(prompt_cfg["responder_meta_prompt"])

    llm1_prompt = render_template(
        prompt_cfg["responder_first_prompt"],
        question=question,
        retrieved_context=ctx,
        round_id="1",
        debate_history="No previous rounds.",
    )
    llm1.add_event(llm1_prompt)
    llm1_raw = llm1.ask(think=False)

    payload, short_answer, parse_error, parse_reason = parse_answer_payload(llm1_raw)

    out: Dict[str, Any] = {
        "retrieved_context": ctx,
        "candidates": [
            _candidate_for_output(
                round_id=1,
                raw_text=llm1_raw,
                payload=payload,
                short_answer=short_answer,
                parse_error=parse_error,
                parse_reason=parse_reason,
            )
        ],
        "rounds": [],
        "judge_reviews": [],
        "judge_raw": None,
        "judge": {},
        "judge_parse_error": False,
        "judge_parse_reason": "",
        "judge_summary": "",
        "final_selected_candidate_id": 1,
        "final_answer": short_answer,
        "final_answer_normalized": normalize_answer_text(short_answer),
        "stop_reason": "single_round_internal_only",
        "debate_rounds_completed": 1,
        "models": {
            "responder": run_cfg.responder_model,
            "critic": None,
            "judge": None,
        },
    }
    if include_prompts:
        out["prompts"] = {
            "llm1_meta": prompt_cfg.get("responder_meta_prompt", ""),
            "llm1_first": llm1_prompt,
        }
    return out


def run_multi_round_debate_qa(
    *,
    question: str,
    retrieved_context: str,
    prompt_cfg: Dict[str, Any],
    run_cfg: RunConfig,
    include_prompts: bool = False,
) -> Dict[str, Any]:
    ctx = (retrieved_context or "").strip()
    rounds: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    judge_reviews: List[Dict[str, Any]] = []
    prompt_records: List[Dict[str, Any]] = []

    llm1 = _make_agent(
        model_name=run_cfg.responder_model,
        name="LLM1",
        temperature=run_cfg.temperature,
        sleep_time=run_cfg.sleep_time,
        api_key=run_cfg.api_key,
        base_url=run_cfg.responder_base_url,
        hf_device=run_cfg.responder_device,
    )
    llm1.set_meta_prompt(prompt_cfg["responder_meta_prompt"])
    llm1_first_prompt = render_template(
        prompt_cfg["responder_first_prompt"],
        question=question,
        retrieved_context=ctx,
        round_id="1",
        debate_history="No previous rounds.",
    )
    llm1.add_event(llm1_first_prompt)
    llm1_raw = llm1.ask(think=False)
    llm1_payload, llm1_answer, llm1_parse_error, llm1_parse_reason = parse_answer_payload(llm1_raw)
    current_candidate = _candidate_for_output(
        round_id=1,
        raw_text=llm1_raw,
        payload=llm1_payload,
        short_answer=llm1_answer,
        parse_error=llm1_parse_error,
        parse_reason=llm1_parse_reason,
    )
    candidates.append(current_candidate)
    if include_prompts:
        prompt_records.append({"agent": "LLM1", "round_id": 1, "prompt": llm1_first_prompt})

    stop_reason = f"max_rounds_{run_cfg.max_rounds}_reached"
    final_judge_raw = ""
    final_judge_payload: Dict[str, Any] = {}
    final_judge_parse_error = False
    final_judge_parse_reason = ""
    final_selected_candidate_id: Optional[int] = None
    final_answer = ""
    final_selection_reason = ""
    final_summary = ""

    while True:
        current_round_id = int(current_candidate["round_id"])

        llm2 = _make_agent(
            model_name=run_cfg.critic_model,
            name="LLM2",
            temperature=run_cfg.temperature,
            sleep_time=run_cfg.sleep_time,
            api_key=run_cfg.api_key,
            base_url=run_cfg.critic_base_url,
        )
        llm2.set_meta_prompt(prompt_cfg["critic_meta_prompt"])
        critic_prompt = render_template(
            prompt_cfg["critic_prompt"],
            question=question,
            retrieved_context=ctx,
            round_id=str(current_round_id),
            debate_history=_debate_history_text(rounds),
            current_candidate_json=json.dumps(current_candidate["payload"] or {"short_answer": current_candidate["short_answer"]}, ensure_ascii=False),
            current_candidate_raw=current_candidate["raw"],
        )
        llm2.add_event(critic_prompt)
        llm2_raw = llm2.ask(think=False)
        critic_payload, critic_parse_error, critic_parse_reason = parse_critic_payload(llm2_raw)

        round_record: Dict[str, Any] = {
            "round_id": current_round_id,
            "candidate": current_candidate,
            "critic_raw": llm2_raw,
            "critic": critic_payload,
            "critic_parse_error": critic_parse_error,
            "critic_parse_reason": critic_parse_reason,
        }
        if include_prompts:
            prompt_records.append({"agent": "LLM2", "round_id": current_round_id, "prompt": critic_prompt})

        llm3_review = _make_agent(
            model_name=run_cfg.judge_model,
            name=f"LLM3-Review-R{current_round_id}",
            temperature=run_cfg.temperature,
            sleep_time=run_cfg.sleep_time,
            api_key=run_cfg.api_key,
            base_url=run_cfg.judge_base_url,
        )
        llm3_review.set_meta_prompt(prompt_cfg["judge_meta_prompt"])
        judge_review_prompt = render_template(
            prompt_cfg["judge_review_prompt"],
            question=question,
            retrieved_context=ctx,
            round_id=str(current_round_id),
            max_rounds=str(run_cfg.max_rounds),
            debate_history=_debate_history_text(rounds),
            current_candidate_json=json.dumps(current_candidate["payload"] or {"short_answer": current_candidate["short_answer"]}, ensure_ascii=False),
            current_candidate_raw=current_candidate["raw"],
            critic_json=json.dumps(critic_payload, ensure_ascii=False),
            candidate_catalog=_candidate_catalog_text(candidates),
        )
        llm3_review.add_event(judge_review_prompt)
        judge_review_raw = llm3_review.ask(think=False)
        judge_review_payload, judge_review_parse_error, judge_review_parse_reason = parse_judge_payload(judge_review_raw)

        round_record["judge_review_raw"] = judge_review_raw
        round_record["judge_review"] = judge_review_payload
        round_record["judge_review_parse_error"] = judge_review_parse_error
        round_record["judge_review_parse_reason"] = judge_review_parse_reason
        rounds.append(round_record)
        judge_reviews.append(
            {
                "round_id": current_round_id,
                "judge_review_raw": judge_review_raw,
                "judge_review": judge_review_payload,
                "judge_review_parse_error": judge_review_parse_error,
                "judge_review_parse_reason": judge_review_parse_reason,
            }
        )
        if include_prompts:
            prompt_records.append({"agent": f"LLM3-Review-R{current_round_id}", "round_id": current_round_id, "prompt": judge_review_prompt})

        should_stop, stop_detail = _judge_review_requests_stop(judge_review_payload, judge_review_parse_error)
        max_reached = len(candidates) >= int(run_cfg.max_rounds)

        if should_stop or max_reached:
            llm3_final = _make_agent(
                model_name=run_cfg.judge_model,
                name="LLM3-Final",
                temperature=run_cfg.temperature,
                sleep_time=run_cfg.sleep_time,
                api_key=run_cfg.api_key,
                base_url=run_cfg.judge_base_url,
            )
            llm3_final.set_meta_prompt(prompt_cfg["judge_meta_prompt"])
            judge_finalize_prompt = render_template(
                prompt_cfg["judge_finalize_prompt"],
                question=question,
                retrieved_context=ctx,
                max_rounds=str(run_cfg.max_rounds),
                stop_reason=(stop_detail if should_stop else f"max_rounds_{run_cfg.max_rounds}_reached"),
                debate_history=_debate_history_text(rounds),
                candidate_catalog=_candidate_catalog_text(candidates),
                latest_judge_review_json=json.dumps(judge_review_payload, ensure_ascii=False),
            )
            llm3_final.add_event(judge_finalize_prompt)
            final_judge_raw = llm3_final.ask(think=False)
            final_judge_payload, final_judge_parse_error, final_judge_parse_reason = parse_judge_payload(final_judge_raw)
            (
                final_selected_candidate_id,
                final_answer,
                final_selection_reason,
                final_summary,
            ) = _finalize_from_judge_payload(final_judge_payload, candidates)

            stop_reason = stop_detail if should_stop else f"max_rounds_{run_cfg.max_rounds}_reached"
            if include_prompts:
                prompt_records.append({"agent": "LLM3-Final", "round_id": current_round_id, "prompt": judge_finalize_prompt})
            break

        next_round_id = len(candidates) + 1
        llm1_rev = _make_agent(
            model_name=run_cfg.responder_model,
            name=f"LLM1-R{next_round_id}",
            temperature=run_cfg.temperature,
            sleep_time=run_cfg.sleep_time,
            api_key=run_cfg.api_key,
            base_url=run_cfg.responder_base_url,
            hf_device=run_cfg.responder_device,
        )
        llm1_rev.set_meta_prompt(prompt_cfg["responder_meta_prompt"])
        revision_prompt = render_template(
            prompt_cfg["responder_revision_prompt"],
            question=question,
            retrieved_context=ctx,
            round_id=str(next_round_id),
            debate_history=_debate_history_text(rounds),
            current_candidate_json=json.dumps(current_candidate["payload"] or {"short_answer": current_candidate["short_answer"]}, ensure_ascii=False),
            critic_json=json.dumps(critic_payload, ensure_ascii=False),
            judge_review_json=json.dumps(judge_review_payload, ensure_ascii=False),
        )
        llm1_rev.add_event(revision_prompt)
        revision_raw = llm1_rev.ask(think=False)
        revision_payload, revision_answer, revision_parse_error, revision_parse_reason = parse_answer_payload(revision_raw)
        revision_candidate = _candidate_for_output(
            round_id=next_round_id,
            raw_text=revision_raw,
            payload=revision_payload,
            short_answer=revision_answer,
            parse_error=revision_parse_error,
            parse_reason=revision_parse_reason,
        )

        rounds[-1]["revision_candidate"] = revision_candidate
        candidates.append(revision_candidate)
        current_candidate = revision_candidate
        if include_prompts:
            prompt_records.append({"agent": f"LLM1-R{next_round_id}", "round_id": next_round_id, "prompt": revision_prompt})

    out: Dict[str, Any] = {
        "retrieved_context": ctx,
        "candidates": candidates,
        "rounds": rounds,
        "judge_reviews": judge_reviews,
        "judge_raw": final_judge_raw,
        "judge": final_judge_payload,
        "judge_parse_error": final_judge_parse_error,
        "judge_parse_reason": final_judge_parse_reason,
        "judge_summary": final_summary,
        "final_selected_candidate_id": final_selected_candidate_id,
        "final_answer": final_answer,
        "final_answer_normalized": normalize_answer_text(final_answer),
        "final_selection_reason": final_selection_reason,
        "stop_reason": stop_reason,
        "debate_rounds_completed": len(candidates),
        "models": {
            "responder": run_cfg.responder_model,
            "critic": run_cfg.critic_model,
            "judge": run_cfg.judge_model,
        },
    }
    if include_prompts:
        out["prompts"] = {
            "llm1_meta": prompt_cfg.get("responder_meta_prompt", ""),
            "llm2_meta": prompt_cfg.get("critic_meta_prompt", ""),
            "llm3_meta": prompt_cfg.get("judge_meta_prompt", ""),
            "records": prompt_records,
        }
    return out
