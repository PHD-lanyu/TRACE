#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

SYSTEM_PROMPT = """You are a helpful assistant. After the user asks a question, you first think carefully and then give the answer.
When responding, please keep the following points in mind:
 - The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.
 - Output your final answer directly between the tag <answer> </answer> without any intermediate steps.
Here is an example:
Question:
what is the capital of China?
<think> reasoning process here </think>
<answer> BeiJing </answer>"""

SOURCE_LABEL_TO_ID = {
    "internal": 0,  # rely on parametric/internal fallback under misleading retrieval
    "external": 1,  # use reliable retrieved context
    "both": 2,      # both internal and external are reliable; answering is preferred
}

# train_hf.py also accepts this mapping. Kept here for stats/readability.
NEG_TYPE_TO_ID = {
    "none": 0,
    "final": 2,
    "internal": 3,
    "candidate": 5,
    "context": 6,
    "drift": 7,
    # v21.3: answer-boundary negatives such as
    # "First Lady" vs "First Lady of the United States".
    "boundary": 8,
}

# Internal construction constants. The public method hyperparameters are only
# target_internal_mass : target_external_mass : target_both_mass. Other values
# mirror the v21.3 reproduction setup and are fixed for cleaner open-source use.
DATA_CONSTRUCTION_CONSTANTS = {
    "include_unsupported_correct": False,
    "max_wrong_candidates_per_stage": 2,
    "prefer_standard_gold_answer": True,
    "use_boundary_mismatch_negatives": True,
    "internal_base_weight": 1.4,
    "external_base_weight": 1.0,
    "both_base_weight": 0.65,
    "clean_ce_weight": 1.0,
    "hard_negative_weight": 1.0,
    "answer_boundary_weight": 1.4,
    "anti_drift_weight": 2.2,
    "context_resistance_weight": 2.0,
    "context_usage_weight": 1.0,
    "stage3_internal_weight": 1.6,
    "final_wrong_weight": 1.3,
    "final_overrode_correct_weight": 1.7,
    "balance_sources": True,
    "min_balance_multiplier": 0.25,
    "max_balance_multiplier": 4.0,
}

BAD_ANSWERS = {
    "", ".", ",", ":", ";", "{", "}", "answer", "final answer", "unknown answer",
    "n/a", "na", "none", "null",
}

REFUSAL_HINTS = (
    "i don't know", "i do not know", "cannot answer", "can't answer", "unknown",
    "not enough information", "insufficient information", "beyond the scope",
)


def clean_ws(text: Any) -> str:
    s = str(text or "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def extract_tagged_answer(text: Any) -> str:
    s = clean_ws(text)
    m = re.search(r"(?is)<answer>\s*(.*?)\s*</answer>", s)
    if m:
        return clean_ws(m.group(1))
    return s


def sanitize_answer(text: Any) -> str:
    s = extract_tagged_answer(text)
    s = re.sub(r"(?is)^.*?(?:short_answer|final answer|answer)\s*[:：]\s*", "", s)
    s = clean_ws(s)
    # Keep first non-empty line. Long rationales must not become answer targets.
    lines = [x.strip() for x in s.split("\n") if x.strip()]
    if lines:
        s = lines[0]
    # Strip surrounding quotes/brackets while preserving answer content.
    s = s.strip().strip('"').strip("'").strip()
    return s


def normalize(text: Any) -> str:
    s = sanitize_answer(text).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def token_set(s: str) -> set:
    return set(x for x in normalize(s).split() if x)


def is_bad_answer(ans: Any) -> bool:
    s = sanitize_answer(ans)
    low = normalize(s)
    if low in BAD_ANSWERS:
        return True
    if len(low) <= 1:
        return True
    if len(low.split()) == 1 and len(low) <= 2:
        # Keeps multiple-choice A/B/C if gold is option-like handled separately by match;
        # for negatives/positives this fragment is too risky.
        return True
    return False


def is_refusal_like(ans: Any) -> bool:
    low = sanitize_answer(ans).lower()
    return any(h in low for h in REFUSAL_HINTS)


def gold_aliases(ex: Dict[str, Any], stage: Dict[str, Any]) -> List[str]:
    ev = stage.get("evaluation") or {}
    raw = ev.get("gold_answers") or ex.get("gold_answers") or []
    if not isinstance(raw, list):
        raw = [raw]
    matched = ev.get("matched_gold_answer")
    if matched:
        raw = [matched] + raw
    out, seen = [], set()
    for g in raw:
        sg = sanitize_answer(g)
        ng = normalize(sg)
        if sg and ng and ng not in seen:
            out.append(sg)
            seen.add(ng)
    return out


def is_abbrev_like_alias(text: Any) -> bool:
    """Heuristic for aliases that are valid for eval but poor SFT targets."""
    raw = sanitize_answer(text)
    if not raw:
        return True
    # Keep normal short words like "law" or "Euro"; penalize mostly uppercase/symbol aliases.
    compact = re.sub(r"[^A-Za-z0-9]+", "", raw)
    if not compact:
        return True
    letters = re.sub(r"[^A-Za-z]+", "", raw)
    if len(compact) <= 4 and raw.upper() == raw and any(c.isalpha() for c in raw):
        return True
    if len(compact) <= 3 and any(ch in raw for ch in [".", "-", "_", "$", "#"]):
        return True
    if len(compact) <= 2:
        return True
    return False


def standard_alias_score(alias: str, index: int) -> float:
    """Rank gold aliases as training targets.

    The first alias in these datasets is usually the canonical answer, so we
    prefer it unless it is an abbreviation/symbol. This is intentionally more
    conservative than simply selecting the longest alias.
    """
    a = sanitize_answer(alias)
    n = normalize(a)
    if not a or not n or is_bad_answer(a):
        return -1e9
    toks = n.split()
    score = 0.0
    if index == 0:
        score += 40.0
    if len(toks) >= 2:
        score += 12.0
    if len(toks) >= 3:
        score += 4.0
    score += min(len(a), 80) * 0.10
    if is_abbrev_like_alias(a):
        score -= 120.0
    # Extremely long historical/technical aliases are valid but often awkward
    # generation targets. Keep them only when they are the canonical first alias.
    if len(toks) > 8:
        score -= (len(toks) - 8) * 6.0
    return score


def choose_standard_gold_alias(aliases: Iterable[str]) -> str:
    cleaned = []
    seen = set()
    for a in aliases:
        sa = sanitize_answer(a)
        na = normalize(sa)
        if sa and na and na not in seen:
            cleaned.append(sa)
            seen.add(na)
    if not cleaned:
        return ""
    best_i, best_s = max(enumerate(cleaned), key=lambda x: standard_alias_score(x[1], x[0]))
    return best_s


def choose_gold(ex: Dict[str, Any], stage: Dict[str, Any]) -> str:
    aliases = gold_aliases(ex, stage)
    std = choose_standard_gold_alias(aliases)
    return std if std else sanitize_answer(stage.get("final_answer") or "")


def answer_exactly_matches_gold(ans: Any, aliases: Iterable[str]) -> bool:
    a = normalize(ans)
    if not a:
        return False
    return any(a == normalize(g) for g in aliases if normalize(g))


def answer_boundary_mismatch(ans: Any, aliases: Iterable[str]) -> bool:
    """True when an answer is close to gold but not an exact alias.

    Examples: "First Lady" vs "First Lady of the United States",
    "Delaware Fightin' Blue Hens" vs "... football",
    or "London, Buckingham Palace" vs "London".
    """
    if answer_exactly_matches_gold(ans, aliases):
        return False
    return answer_matches_gold(ans, aliases)


def standardize_positive_answer(pos: Any, aliases: Iterable[str], prefer_standard: bool = True) -> str:
    p = sanitize_answer(pos)
    if not prefer_standard:
        return p
    if answer_matches_gold(p, aliases):
        std = choose_standard_gold_alias(aliases)
        if std:
            return std
    return p


def answer_matches_gold(ans: Any, aliases: Iterable[str]) -> bool:
    a = normalize(ans)
    if not a:
        return False
    for g in aliases:
        gg = normalize(g)
        if not gg:
            continue
        if a == gg:
            return True
        # Safe containment for entity-like answers. Avoid one-letter option false positives.
        if len(gg) >= 4 and (f" {gg} " in f" {a} " or f" {a} " in f" {gg} "):
            return True
        at, gt = token_set(a), token_set(gg)
        if gt and len(gt) >= 2 and gt.issubset(at):
            return True
        if at and gt:
            jac = len(at & gt) / max(1, len(at | gt))
            if jac >= 0.82:
                return True
    return False


def weak_negative(ans: Any, aliases: Iterable[str], *, allow_boundary_mismatch: bool = False) -> bool:
    """Return True when an answer is unusable as a negative.

    Exact gold aliases should never be negative. In v21.3, however,
    boundary mismatches can optionally be used as negatives to teach answer
    completeness, e.g. "First Lady" when gold is "First Lady of the United States".
    """
    if is_bad_answer(ans) or is_refusal_like(ans):
        return True
    if answer_exactly_matches_gold(ans, aliases):
        return True
    if answer_matches_gold(ans, aliases):
        return not allow_boundary_mismatch
    return False


def candidate_answer(cand: Dict[str, Any]) -> str:
    payload = cand.get("payload") if isinstance(cand, dict) else {}
    if isinstance(payload, dict):
        ans = sanitize_answer(payload.get("short_answer") or "")
        if ans:
            return ans
    return sanitize_answer((cand or {}).get("short_answer") or (cand or {}).get("raw") or "")


def stage_final_answer(stage: Dict[str, Any]) -> str:
    sid = stage.get("final_selected_candidate_id")
    try:
        sid_i = int(sid)
    except Exception:
        sid_i = None
    if sid_i is not None:
        for cand in stage.get("candidates") or []:
            try:
                if int(cand.get("round_id", -999)) == sid_i:
                    ca = candidate_answer(cand)
                    if ca:
                        return ca
            except Exception:
                pass
    return sanitize_answer(stage.get("final_answer") or "")


def get_internal_answer(ex: Dict[str, Any]) -> Tuple[str, bool, bool]:
    """Return answer, content-correct-ish flag, parse_error flag."""
    st1 = ex.get("stage1") or {}
    aliases = gold_aliases(ex, st1)
    cands = st1.get("candidates") or []
    parse_error = bool(st1.get("judge_parse_error"))
    ans = sanitize_answer(st1.get("final_answer") or "")
    for cand in cands[:1]:
        parse_error = parse_error or bool(cand.get("parse_error"))
        ca = candidate_answer(cand)
        if ca and not is_bad_answer(ca):
            ans = ca
            break
    field_correct = bool(ex.get("stage1_internal_correct"))
    content_correct = field_correct or answer_matches_gold(ans, aliases)
    return ans, content_correct, parse_error


def all_stage_candidates(stage: Dict[str, Any], aliases: Iterable[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    correct, wrong = [], []
    seen = set()
    for cand in stage.get("candidates") or []:
        ans = candidate_answer(cand)
        key = normalize(ans)
        if not key or key in seen:
            continue
        seen.add(key)
        item = {
            "round_id": cand.get("round_id"),
            "answer": ans,
            "parse_error": bool(cand.get("parse_error")),
            "used_context": clean_ws(((cand.get("payload") or {}).get("used_context") if isinstance(cand.get("payload"), dict) else "")),
            "confidence": (cand.get("payload") or {}).get("confidence") if isinstance(cand.get("payload"), dict) else None,
        }
        exact_match = answer_exactly_matches_gold(ans, aliases)
        soft_match = answer_matches_gold(ans, aliases)
        boundary = bool(soft_match and not exact_match)
        item["exact_gold_match"] = exact_match
        item["boundary_mismatch"] = boundary
        if soft_match:
            # Use near-correct candidates as evidence that the debate reached
            # the right entity, but later standardize the positive target to
            # the canonical gold alias. Boundary-mismatched answers can also
            # be used as negatives when enabled.
            correct.append(item)
            if boundary:
                wrong.append(item)
        elif not weak_negative(ans, aliases):
            wrong.append(item)
    return correct, wrong


def make_user_prompt(question: str, retrieved_context: str) -> str:
    # Match the evaluation-time RAG input as much as possible. Do not include the debate trace.
    return f"Retrieved information:\n{clean_ws(retrieved_context)}\n\nQuestion:\n{clean_ws(question)}"


def answer_template(ans: str) -> str:
    return f"<think>\n</think>\n<answer>{clean_ws(ans)}</answer>"



def infer_source_label(internal_correct: bool, external_correct: bool, stage_correct: bool, has_correct_candidate: bool, include_unsupported_correct: bool) -> Optional[str]:
    if external_correct and internal_correct:
        return "both"
    if external_correct and not internal_correct:
        return "external"
    if (not external_correct) and internal_correct:
        return "internal"
    # Both internal and retrieval are unreliable. Do not create abstain/refusal records.
    # Optionally keep rare examples where the debate found a correct answer despite both flags.
    if include_unsupported_correct and (stage_correct or has_correct_candidate):
        return "internal"
    return None


def base_weight_for(args: argparse.Namespace, source: str, module: str, stage_key: str, final_correct: bool, final_overrode_correct: bool) -> float:
    w = 1.0
    w *= float(getattr(args, f"{source}_base_weight"))
    if module == "anti_drift":
        w *= float(args.anti_drift_weight)
    elif module == "context_resistance":
        w *= float(args.context_resistance_weight)
    elif module == "context_usage":
        w *= float(args.context_usage_weight)
    elif module == "clean_ce":
        w *= float(args.clean_ce_weight)
    elif module == "hard_negative":
        w *= float(args.hard_negative_weight)
    elif module == "answer_boundary":
        w *= float(args.answer_boundary_weight)
    if stage_key == "stage3" and source == "internal":
        w *= float(args.stage3_internal_weight)
    if final_correct is False:
        w *= float(args.final_wrong_weight)
    if final_overrode_correct:
        w *= float(args.final_overrode_correct_weight)
    return w


def build_record(
    *,
    ex: Dict[str, Any],
    stage_key: str,
    stage: Dict[str, Any],
    source: str,
    module: str,
    question: str,
    retrieved_context: str,
    gold: str,
    pos: str,
    neg: str = "",
    neg_type: str = "none",
    sample_weight: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "example_id": ex.get("example_id"),
        "stage_name": stage_key,
        "module": module,
        "source_label": source,
        "source_label_id": SOURCE_LABEL_TO_ID[source],
        "neg_type": neg_type,
        "negative_type": neg_type,
        "category": f"{source}_{module}",
        "question": question,
        "retrieved_context": retrieved_context,
        "gold_answer": gold,
        "stage_external_correct": bool(stage.get("external_correct")),
        "stage_answer_correct": bool((stage.get("evaluation") or {}).get("correct")) if isinstance((stage.get("evaluation") or {}).get("correct"), bool) else None,
        "case_id": stage.get("case_id"),
        "case_label": stage.get("case_label"),
        "sample_weight": float(sample_weight),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_prompt(question, retrieved_context)},
        ],
        "pos_answer": answer_template(pos) if pos else "",
        "neg_answer": answer_template(neg) if neg else "",
        "metadata": metadata or {},
    }


def convert_stage(ex: Dict[str, Any], stage_key: str, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Counter]:
    stage = ex.get(stage_key) or {}
    if not stage:
        return [], Counter({"missing_stage": 1})

    question = clean_ws(ex.get("question") or "")
    context = clean_ws(stage.get("retrieved_context") or "")
    aliases = gold_aliases(ex, stage)
    gold = choose_gold(ex, stage)
    if not question or not gold or is_bad_answer(gold):
        return [], Counter({"bad_question_or_gold": 1})

    internal_ans, internal_correct, internal_parse_error = get_internal_answer(ex)
    external_correct = bool(stage.get("external_correct"))
    final_ans = stage_final_answer(stage)
    # Use strict/evaluator correctness for final answers. Soft containment
    # matches such as "First Lady" vs "First Lady of the United States"
    # are useful signals, but should not make the final answer positive.
    final_exact_correct = answer_exactly_matches_gold(final_ans, aliases)
    final_boundary_mismatch = answer_boundary_mismatch(final_ans, aliases)
    final_correct = final_exact_correct or bool((stage.get("evaluation") or {}).get("correct"))
    correct_cands, wrong_cands = all_stage_candidates(stage, aliases)
    has_correct_candidate = bool(correct_cands)
    final_overrode_correct = bool(has_correct_candidate and not final_correct)

    source = infer_source_label(
        internal_correct=internal_correct,
        external_correct=external_correct,
        stage_correct=final_correct,
        has_correct_candidate=has_correct_candidate,
        include_unsupported_correct=bool(args.include_unsupported_correct),
    )
    if source is None:
        return [], Counter({"skipped_both_wrong_no_abstain": 1})

    # Teacher answer is selected by reliability, not by final debate answer.
    # v21.3 then canonicalizes gold-matching positives to the standard gold
    # alias, preventing short candidates such as "First Lady" from becoming
    # CE targets when the evaluator expects "First Lady of the United States".
    if source == "internal" and internal_correct and not is_bad_answer(internal_ans):
        pos = sanitize_answer(internal_ans)
    elif correct_cands:
        pos = sanitize_answer(correct_cands[0]["answer"])
    else:
        pos = gold
    pos_before_standardization = pos
    pos = standardize_positive_answer(pos, aliases, prefer_standard=bool(args.prefer_standard_gold_answer))
    if not pos or is_bad_answer(pos):
        pos = gold

    meta = {
        "internal_answer": internal_ans,
        "internal_correct": internal_correct,
        "internal_parse_error": internal_parse_error,
        "external_correct": external_correct,
        "final_answer": final_ans,
        "final_correct": final_correct,
        "final_exact_correct": final_exact_correct,
        "final_boundary_mismatch": final_boundary_mismatch,
        "standard_gold_answer": gold,
        "pos_answer_before_standardization": pos_before_standardization,
        "pos_answer_standardized": pos != sanitize_answer(pos_before_standardization),
        "has_correct_candidate": has_correct_candidate,
        "first_correct_candidate": correct_cands[0] if correct_cands else None,
        "wrong_candidate_count": len(wrong_cands),
    }

    records: List[Dict[str, Any]] = []
    stats = Counter()

    def add(module: str, neg: str = "", neg_type: str = "none", weight: Optional[float] = None, extra_meta: Optional[Dict[str, Any]] = None) -> None:
        if neg and weak_negative(neg, aliases, allow_boundary_mismatch=bool(args.use_boundary_mismatch_negatives)):
            stats["dropped_weak_negative"] += 1
            return
        m = dict(meta)
        if extra_meta:
            m.update(extra_meta)
        w = weight if weight is not None else base_weight_for(args, source, module, stage_key, final_correct, final_overrode_correct)
        records.append(build_record(
            ex=ex,
            stage_key=stage_key,
            stage=stage,
            source=source,
            module=module,
            question=question,
            retrieved_context=context,
            gold=gold,
            pos=pos,
            neg=neg,
            neg_type=neg_type,
            sample_weight=w,
            metadata=m,
        ))
        stats["kept"] += 1
        stats[f"source_{source}"] += 1
        stats[f"module_{module}"] += 1
        stats[f"neg_{neg_type}"] += 1
        stats[f"case_{stage.get('case_id')}"] += 1

    # Clean CE always exists for answerable/reliable-source cases.
    add("clean_ce")

    # If the final debate answer is wrong, use it as hard negative, but only answer-span UL in train_hf.py.
    # Boundary mismatches get a dedicated neg_type so train_hf.py can include them
    # in the internal final-like margin while stats stay interpretable.
    if not final_correct and not weak_negative(final_ans, aliases, allow_boundary_mismatch=bool(args.use_boundary_mismatch_negatives)):
        module = "anti_drift" if has_correct_candidate else ("context_resistance" if source == "internal" else "context_usage")
        neg_type = "boundary" if final_boundary_mismatch else ("drift" if has_correct_candidate else "final")
        add(module, neg=final_ans, neg_type=neg_type, extra_meta={"final_boundary_negative": final_boundary_mismatch})

    # Candidate-level hard negatives: keep a few representative wrong candidates, not every noisy turn.
    max_wrong = int(args.max_wrong_candidates_per_stage)
    used_neg = {normalize(final_ans)} if final_ans else set()
    for cand in wrong_cands[:max_wrong]:
        ca = cand["answer"]
        key = normalize(ca)
        if not key or key in used_neg:
            continue
        used_neg.add(key)
        if bool(cand.get("boundary_mismatch")):
            module, neg_type = "answer_boundary", "boundary"
        elif source == "internal" and external_correct is False:
            module, neg_type = "context_resistance", "context"
        elif source == "external":
            module, neg_type = "context_usage", "candidate"
        else:
            module, neg_type = "hard_negative", "candidate"
        add(module, neg=ca, neg_type=neg_type, extra_meta={"negative_candidate": cand})

    # If external context is correct and internal answer is wrong, explicitly teach context to outrank internal prior.
    if source == "external" and not weak_negative(internal_ans, aliases):
        add("context_usage", neg=internal_ans, neg_type="internal")

    return records, stats



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=str, required=True)
    ap.add_argument("--out_path", type=str, required=True)
    ap.add_argument("--out_stats", type=str, default=None)
    ap.add_argument("--max_examples", type=int, default=None)

    # Construction controls.

    # Base weights. These are later globally balanced by source mass.

    # Global source balancing. This addresses the hard-dataset skew where most records otherwise teach external reliance.
    ap.add_argument("--target_internal_mass", type=float, default=0.45)
    ap.add_argument("--target_external_mass", type=float, default=0.40)
    ap.add_argument("--target_both_mass", type=float, default=0.15)
    args = ap.parse_args()
    for _name, _value in DATA_CONSTRUCTION_CONSTANTS.items():
        setattr(args, _name, _value)

    with open(args.in_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    rows = obj.get("results", obj) if isinstance(obj, dict) else obj
    if args.max_examples is not None:
        rows = rows[: int(args.max_examples)]

    # Stream records to avoid holding duplicated long contexts in memory.
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)) or ".", exist_ok=True)
    tmp_path = args.out_path + ".tmp_prebalance"
    stats = Counter()
    pre_mass = Counter()
    total_records = 0
    with open(tmp_path, "w", encoding="utf-8") as ftmp:
        for ex in rows:
            for stage_key in ("stage2", "stage3"):
                recs, ctr = convert_stage(ex, stage_key, args)
                stats.update(ctr)
                for r in recs:
                    total_records += 1
                    pre_mass[r.get("source_label", "")] += float(r.get("sample_weight", 1.0))
                    ftmp.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compute global source-balance multipliers from the streamed pre-balance mass.
    if args.balance_sources and total_records > 0:
        target = {
            "internal": float(args.target_internal_mass),
            "external": float(args.target_external_mass),
            "both": float(args.target_both_mass),
        }
        available = {s: target[s] for s in target if pre_mass.get(s, 0.0) > 0.0}
        total_target = sum(available.values()) or 1.0
        available = {s: v / total_target for s, v in available.items()}
        total_current = sum(pre_mass.values()) or 1.0
        multipliers = {}
        for s, mass in pre_mass.items():
            desired = available.get(s, 0.0) * total_current
            mult = 0.0 if desired <= 0 else desired / max(1e-8, float(mass))
            mult = max(float(args.min_balance_multiplier), min(float(args.max_balance_multiplier), mult))
            multipliers[s] = mult
        balance_summary = {
            "balanced": True,
            "target_mass": target,
            "available_target_mass": available,
            "pre_balance_mass": dict(pre_mass),
            "multipliers": multipliers,
            "total_records": total_records,
        }
    else:
        multipliers = defaultdict(lambda: 1.0)
        balance_summary = {"balanced": False, "total_records": total_records, "pre_balance_mass": dict(pre_mass)}

    with open(tmp_path, "r", encoding="utf-8") as fin, open(args.out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            r = json.loads(line)
            src = r.get("source_label", "")
            old_w = float(r.get("sample_weight", 1.0))
            r["pre_balance_weight"] = old_w
            r["sample_weight"] = old_w * float(multipliers.get(src, 1.0))
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    summary = {
        "total_records": total_records,
        "stats": dict(stats),
        "balance": balance_summary,
        "source_label_map": SOURCE_LABEL_TO_ID,
        "neg_type_map": NEG_TYPE_TO_ID,
        "notes": {
            "version": "v21_3_alias_complete_answercentric",
            "no_abstain_records": True,
            "prompt_policy": "Only Retrieved information + Question are included. Debate trace is not used as teacher input.",
            "positive_policy": "Positive answer is selected by source reliability and then canonicalized to the standard/full gold alias when it matches gold, avoiding short debate candidates as CE targets.",
            "negative_policy": "Wrong final answers, wrong candidates, and answer-boundary near-misses become answer-span UL negatives; train_hf.py masks UL to <answer> core by default.",
            "balance_policy": "Global source-mass balancing prevents the hard dataset from overwhelmingly teaching external-context reliance.",
        },
    }
    if args.out_stats:
        with open(args.out_stats, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
