import argparse
import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List

import torch
from vllm import LLM, SamplingParams

from data_process.choice import choice
from data_process.conflict_qa import conflict, conflict_mix
from data_process.hotpot import hotpotqa
from data_process.musique import musique
from data_process.processing_knowledge import calculate_metrics, prepare_prompts, process_outputs

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. After the user asks a question, you first think carefully and then give the answer.
When responding, please keep the following points in mind:
 - The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.
 - Output your final answer directly between the tag <answer> </answer> without any intermediate steps.
Here is an example:
Question:
what is the capital of China?
<think> reasoning process here </think>
<answer> Beijing </answer>"""

MAX_RESULTS = 5
MAX_MODEL_LEN = 4096
MAX_TOKENS = 256
TEMPERATURE = 0.0
TOP_P = 1.0
REPETITION_PENALTY = 1.0
GPU_MEMORY_UTILIZATION = 0.85
DTYPE = torch.bfloat16
EVAL_THREADS = 8


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    relative_path: str
    loader: Callable


DATASETS: Dict[str, DatasetSpec] = {
    "hotpotqa": DatasetSpec("hotpotqa", "hotpot/hotpot_dev_fullwiki_v1.json", hotpotqa),
    "2wiki": DatasetSpec("2wiki", "2wiki/dev.json", hotpotqa),
    "musique": DatasetSpec("musique", "MuSiQue/musique_ans_v1.0_dev.jsonl", musique),
    "explainpe": DatasetSpec("explainpe", "explainpe/explainpe_test.xlsx", choice),
    "confiqa_qa": DatasetSpec("confiqa_qa", "conflict_qa/ConFiQA-QA-test.json", conflict),
    "confiqa_mr": DatasetSpec("confiqa_mr", "conflict_qa/ConFiQA-MR-test.json", conflict),
    "confiqa_mc": DatasetSpec("confiqa_mc", "conflict_qa/ConFiQA-MC-test.json", conflict),
    "confiqa_sc": DatasetSpec("confiqa_sc", "conflict_qa/ConFiQA-MIX-test.json", conflict_mix),
}

ALIASES = {
    "qa": ["hotpotqa", "musique", "2wiki"],
    "confiqa": ["confiqa_qa", "confiqa_mr", "confiqa_mc", "confiqa_sc"],
    "all": list(DATASETS.keys()),
}


def parse_bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_arguments():
    parser = argparse.ArgumentParser(description="RAG conflict evaluation with vLLM")
    parser.add_argument("--model", required=True, help="Merged model path or Hugging Face model id")
    parser.add_argument("--data-path", required=True, help="Dataset root directory")
    parser.add_argument("--output-dir", required=True, help="Directory used to save json results")
    parser.add_argument("--datasets", default="hotpotqa", help="Dataset name, alias, or comma-separated dataset list")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rag", type=parse_bool, default=True)
    parser.add_argument("--confiqa-qa-file", default=None)
    return parser.parse_args()


def expand_datasets(selection: str) -> List[str]:
    if selection in ALIASES:
        return ALIASES[selection]
    names = [item.strip() for item in selection.split(",") if item.strip()]
    unknown = [name for name in names if name not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown dataset(s): {', '.join(unknown)}")
    return names


def load_dataset(name: str, data_root: str, confiqa_qa_file: str = None) -> List[Dict]:
    spec = DATASETS[name]
    path = confiqa_qa_file if name == "confiqa_qa" and confiqa_qa_file else os.path.join(data_root, spec.relative_path)
    samples = spec.loader(path, MAX_RESULTS)
    print(f"Loaded {len(samples)} samples from {name}: {path}")
    return samples


def save_result(output_dir: str, dataset_name: str, results: List[Dict], metrics: Dict, args) -> None:
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "results": results,
        "metrics": metrics,
        "config": {
            "model": args.model,
            "data_path": args.data_path,
            "datasets": args.datasets,
            "rag": args.rag,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_results": MAX_RESULTS,
            "max_model_len": MAX_MODEL_LEN,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        },
    }
    with open(os.path.join(output_dir, f"{dataset_name}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    args = parse_arguments()
    args.system_prompt = DEFAULT_SYSTEM_PROMPT
    dataset_names = expand_datasets(args.datasets)
    all_samples = {name: load_dataset(name, args.data_path, args.confiqa_qa_file) for name in dataset_names}
    if not any(all_samples.values()):
        raise RuntimeError("No samples were loaded. Check --data-path and --datasets.")

    print(f"Initializing model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=DTYPE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        device=args.device,
        max_model_len=MAX_MODEL_LEN,
    )
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        repetition_penalty=REPETITION_PENALTY,
    )

    for dataset_name, samples in all_samples.items():
        prompts, metadata = prepare_prompts(dataset_name, samples, args)
        outputs = llm.generate(prompts, sampling_params)
        results = process_outputs(outputs, metadata, EVAL_THREADS)
        for item in results:
            item.pop("document", None)
        metrics = calculate_metrics(results)
        save_result(args.output_dir, dataset_name, results, metrics, args)
        print(f"{dataset_name}: total={len(results)}, accuracy={metrics['accuracy']:.4f}")
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
