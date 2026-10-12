"""Main QA experiment for ConFiQA-style data.

Three stages per example:
1. Stage1: no retrieval, no debate, LLM1 answers from internal knowledge only.
2. Stage2: correct retrieval (orig_context) + multi-round debate + LLM3 judge.
3. Stage3: wrong retrieval (cf_context) + multi-round debate + LLM3 judge.

This script also:
- labels each debated stage into one of 8 user-defined cases
- records full debate traces for later retrieval-material construction
- supports LLM1 on local HuggingFace and LLM2/LLM3 on local Ollama
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from utils.qa_debate import (
    RunConfig,
    answer_is_correct,
    collect_gold_answers,
    run_multi_round_debate_qa,
    run_responder_only_qa,
)


CASE_DEFS = {
    1: {
        "label": "internal_correct_external_correct_answer_correct",
        "internal_correct": True,
        "external_correct": True,
        "answer_correct": True,
    },
    2: {
        "label": "internal_correct_external_correct_answer_wrong",
        "internal_correct": True,
        "external_correct": True,
        "answer_correct": False,
    },
    3: {
        "label": "internal_correct_external_wrong_answer_correct",
        "internal_correct": True,
        "external_correct": False,
        "answer_correct": True,
    },
    4: {
        "label": "internal_correct_external_wrong_answer_wrong",
        "internal_correct": True,
        "external_correct": False,
        "answer_correct": False,
    },
    5: {
        "label": "internal_wrong_external_correct_answer_correct",
        "internal_correct": False,
        "external_correct": True,
        "answer_correct": True,
    },
    6: {
        "label": "internal_wrong_external_correct_answer_wrong",
        "internal_correct": False,
        "external_correct": True,
        "answer_correct": False,
    },
    7: {
        "label": "internal_wrong_external_wrong_answer_correct",
        "internal_correct": False,
        "external_correct": False,
        "answer_correct": True,
    },
    8: {
        "label": "internal_wrong_external_wrong_answer_wrong",
        "internal_correct": False,
        "external_correct": False,
        "answer_correct": False,
    },
}


def load_prompt_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _bool(x: Any) -> bool:
    return bool(x)


def determine_case_id(internal_correct: bool, external_correct: bool, answer_correct: bool) -> int:
    for case_id, meta in CASE_DEFS.items():
        if (
            meta["internal_correct"] == bool(internal_correct)
            and meta["external_correct"] == bool(external_correct)
            and meta["answer_correct"] == bool(answer_correct)
        ):
            return case_id
    raise ValueError("No matching case definition found")


def evaluate_prediction(pred_answer: str, gold_answers: List[str]) -> Dict[str, Any]:
    correct, matched_gold, pred_norm, gold_norms = answer_is_correct(pred_answer, gold_answers)
    return {
        "pred_answer": pred_answer,
        "pred_answer_normalized": pred_norm,
        "gold_answers": gold_answers,
        "gold_answers_normalized": gold_norms,
        "matched_gold_answer": matched_gold,
        "correct": correct,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "ConFiQA QA main experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--input-json", type=str, required=True, help="Path to ConFiQA-style JSON list")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory")
    p.add_argument("--output-json", type=str, default="", help="Default: <output-dir>/qa_results.json")
    p.add_argument("--api-key", type=str, default="", help="Used only when your Ollama/OpenAI endpoint requires auth")

    p.add_argument("--responder-model", type=str, default=os.getenv("RESPONDER_MODEL", "/path/to/Qwen2.5-7B-Instruct"))
    p.add_argument("--critic-model", type=str, default="qwen3:32b")
    p.add_argument("--judge-model", type=str, default="deepseek-r1:8b")

    p.add_argument("--responder-device", type=str, default=os.getenv("HF_DEVICE", "cuda:0"))
    p.add_argument("--critic-base-url", type=str, default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    p.add_argument("--judge-base-url", type=str, default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    p.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0")))
    p.add_argument("--sleep-time", type=float, default=float(os.getenv("SLEEP_TIME", "0")))
    p.add_argument("--max-rounds", type=int, default=5, help="Maximum LLM1 debate rounds (initial answer counts as round 1)")

    # Ollama runtime controls. These are forwarded through environment variables
    # and read by Agent._ollama_chat(). Keeping them here makes the launcher
    # self-contained and avoids accidental 128K context KV-cache explosions.
    p.add_argument("--ollama-num-ctx", type=int, default=int(os.getenv("OLLAMA_NUM_CTX", os.getenv("OLLAMA_CONTEXT_LENGTH", "8192"))))
    p.add_argument("--ollama-num-predict", type=int, default=int(os.getenv("OLLAMA_NUM_PREDICT", "1024")))
    p.add_argument("--ollama-keep-alive", type=str, default=os.getenv("OLLAMA_KEEP_ALIVE", "0"))

    p.add_argument(
        "--prompt-config",
        type=str,
        default=os.getenv("PROMPT_CONFIG", str(Path(__file__).resolve().parent / "utils" / "config4qa_confiqa.json")),
        help="Prompt config JSON path",
    )
    p.add_argument("--max-examples", type=int, default=0, help="If >0, only run the first N examples after slicing")
    p.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    p.add_argument("--end", type=int, default=-1, help="End index (exclusive). -1 means until the end")
    p.add_argument("--quiet", action="store_true", help="Disable per-call printing (sets MAD_VERBOSE=0)")
    p.add_argument(
        "--include-prompts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to include prompts/meta-prompts in the output JSON",
    )

    return p.parse_args()


def build_stage_record(
    *,
    stage_name: str,
    external_correct: bool,
    result: Dict[str, Any],
    gold_answers: List[str],
    internal_correct: bool,
) -> Dict[str, Any]:
    evaluation = evaluate_prediction(result.get("final_answer", ""), gold_answers)
    case_id = determine_case_id(
        internal_correct=internal_correct,
        external_correct=external_correct,
        answer_correct=evaluation["correct"],
    )
    return {
        "stage_name": stage_name,
        "external_correct": external_correct,
        **result,
        "evaluation": evaluation,
        "case_id": case_id,
        "case_label": CASE_DEFS[case_id]["label"],
    }


def main() -> None:
    args = parse_args()

    if args.quiet:
        os.environ["MAD_VERBOSE"] = "0"

    # Make Ollama behavior explicit for every /api/chat call.
    os.environ["OLLAMA_NUM_CTX"] = str(args.ollama_num_ctx)
    os.environ["OLLAMA_NUM_PREDICT"] = str(args.ollama_num_predict)
    os.environ["OLLAMA_KEEP_ALIVE"] = str(args.ollama_keep_alive)

    os.makedirs(args.output_dir, exist_ok=True)
    prompt_cfg = load_prompt_config(args.prompt_config)

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("input-json must be a JSON list")

    start = max(0, int(args.start))
    end = len(data) if int(args.end) < 0 else min(int(args.end), len(data))
    subset = data[start:end]
    if args.max_examples and args.max_examples > 0:
        subset = subset[: args.max_examples]

    run_cfg = RunConfig(
        responder_model=args.responder_model,
        critic_model=args.critic_model,
        judge_model=args.judge_model,
        temperature=float(args.temperature),
        api_key=args.api_key,
        sleep_time=float(args.sleep_time),
        max_rounds=int(args.max_rounds),
        responder_device=args.responder_device,
        critic_base_url=args.critic_base_url,
        judge_base_url=args.judge_base_url,
    )

    results: List[Dict[str, Any]] = []
    case_counts = {str(i): 0 for i in range(1, 9)}
    case_counts_by_stage = {
        "stage2_orig_context": {str(i): 0 for i in range(1, 9)},
        "stage3_cf_context": {str(i): 0 for i in range(1, 9)},
    }

    stage1_correct_count = 0
    stage2_correct_count = 0
    stage3_correct_count = 0

    for idx, ex in enumerate(tqdm(subset, desc="Running ConFiQA QA experiment")):
        example_id = start + idx
        question = str(ex.get("question") or "").strip()
        gold_answers = collect_gold_answers(ex)
        orig_context = str(ex.get("orig_context") or "").strip()
        cf_context = str(ex.get("cf_context") or "").strip()

        stage1_raw = run_responder_only_qa(
            question=question,
            retrieved_context="",
            prompt_cfg=prompt_cfg,
            run_cfg=run_cfg,
            include_prompts=bool(args.include_prompts),
        )
        stage1_eval = evaluate_prediction(stage1_raw.get("final_answer", ""), gold_answers)
        internal_correct = _bool(stage1_eval["correct"])
        if internal_correct:
            stage1_correct_count += 1

        stage1 = {
            "stage_name": "stage1_internal_only",
            **stage1_raw,
            "evaluation": stage1_eval,
        }

        stage2 = build_stage_record(
            stage_name="stage2_orig_context",
            external_correct=True,
            result=run_multi_round_debate_qa(
                question=question,
                retrieved_context=orig_context,
                prompt_cfg=prompt_cfg,
                run_cfg=run_cfg,
                include_prompts=bool(args.include_prompts),
            ),
            gold_answers=gold_answers,
            internal_correct=internal_correct,
        )
        if _bool(stage2["evaluation"]["correct"]):
            stage2_correct_count += 1
        case_counts[str(stage2["case_id"])] += 1
        case_counts_by_stage["stage2_orig_context"][str(stage2["case_id"])] += 1

        stage3 = build_stage_record(
            stage_name="stage3_cf_context",
            external_correct=False,
            result=run_multi_round_debate_qa(
                question=question,
                retrieved_context=cf_context,
                prompt_cfg=prompt_cfg,
                run_cfg=run_cfg,
                include_prompts=bool(args.include_prompts),
            ),
            gold_answers=gold_answers,
            internal_correct=internal_correct,
        )
        if _bool(stage3["evaluation"]["correct"]):
            stage3_correct_count += 1
        case_counts[str(stage3["case_id"])] += 1
        case_counts_by_stage["stage3_cf_context"][str(stage3["case_id"])] += 1

        results.append(
            {
                "example_id": example_id,
                "question": question,
                "gold_answers": gold_answers,
                "orig_context": orig_context,
                "cf_context": cf_context,
                "stage1_internal_correct": internal_correct,
                "stage1": stage1,
                "stage2": stage2,
                "stage3": stage3,
            }
        )

    total_examples = len(results)
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        "input_json": os.path.abspath(args.input_json),
        "output_dir": os.path.abspath(args.output_dir),
        "start": start,
        "end": end,
        "ran": total_examples,
        "models": {
            "responder": run_cfg.responder_model,
            "critic": run_cfg.critic_model,
            "judge": run_cfg.judge_model,
        },
        "devices": {
            "responder_device": run_cfg.responder_device,
            "critic_base_url": run_cfg.critic_base_url,
            "judge_base_url": run_cfg.judge_base_url,
        },
        "prompt_config": os.path.abspath(args.prompt_config),
        "max_rounds": run_cfg.max_rounds,
        "counts": {
            "examples": total_examples,
            "internal_correct_stage1": stage1_correct_count,
            "internal_wrong_stage1": total_examples - stage1_correct_count,
            "stage2_correct": stage2_correct_count,
            "stage2_wrong": total_examples - stage2_correct_count,
            "stage3_correct": stage3_correct_count,
            "stage3_wrong": total_examples - stage3_correct_count,
        },
        "accuracies": {
            "stage1_internal_only": (stage1_correct_count / total_examples) if total_examples else 0.0,
            "stage2_orig_context": (stage2_correct_count / total_examples) if total_examples else 0.0,
            "stage3_cf_context": (stage3_correct_count / total_examples) if total_examples else 0.0,
        },
        "case_counts_overall": case_counts,
        "case_counts_by_stage": case_counts_by_stage,
        "case_definitions": CASE_DEFS,
    }

    out_path = args.output_json.strip() or os.path.join(args.output_dir, "qa_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    retrieval_material_path = os.path.join(args.output_dir, "retrieval_material.jsonl")
    with open(retrieval_material_path, "w", encoding="utf-8") as f:
        for item in results:
            for stage_key in ("stage2", "stage3"):
                stage = item[stage_key]
                record = {
                    "example_id": item["example_id"],
                    "question": item["question"],
                    "stage": stage_key,
                    "case_id": stage["case_id"],
                    "case_label": stage["case_label"],
                    "stage1_internal_correct": item["stage1_internal_correct"],
                    "external_correct": stage["external_correct"],
                    "retrieved_context": stage.get("retrieved_context", ""),
                    "debate_rounds_completed": stage.get("debate_rounds_completed"),
                    "stop_reason": stage.get("stop_reason"),
                    "candidates": stage.get("candidates", []),
                    "rounds": stage.get("rounds", []),
                    "judge": stage.get("judge", {}),
                    "judge_raw": stage.get("judge_raw"),
                    "final_selected_candidate_id": stage.get("final_selected_candidate_id"),
                    "final_answer": stage.get("final_answer"),
                    "evaluation": stage.get("evaluation", {}),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
