from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

from tqdm import tqdm

from data_process.qa_utils import exact_match, extract_answer


def prepare_prompts(dataset_name: str, samples: List[Dict], args) -> Tuple[List[Dict], List[Dict]]:
    prompts, metadata = [], []
    for item in tqdm(samples, desc=f"Preparing {dataset_name} prompts"):
        query = item["passage"] + item["problem"] if args.rag else item["problem"]
        prompt_text = (
            f"<|im_start|>system\n{args.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        prompts.append({"prompt": prompt_text})
        meta = {
            "dataset": dataset_name,
            "question": item["problem"],
            "answer": item["answer"],
            "prompt": prompt_text,
        }
        if "is_cf" in item:
            meta["is_cf"] = item["is_cf"]
        metadata.append(meta)
    return prompts, metadata


def evaluate_prediction(prediction: str, answer: str) -> float:
    extracted = extract_answer(prediction)
    if extracted is None:
        return 0.0
    return 1.0 if exact_match(extracted, answer) else 0.0


def process_outputs(outputs, metadata: List[Dict], max_workers: int) -> List[Dict]:
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, output in enumerate(outputs):
            prediction = output.outputs[0].text.strip()
            futures.append((executor.submit(evaluate_prediction, prediction, metadata[idx]["answer"]), prediction, metadata[idx]))
        for future, prediction, meta in tqdm(futures, desc="Evaluating predictions"):
            accuracy = future.result()
            results.append({
                "question": meta["question"],
                "answer": meta["answer"],
                "prediction": prediction,
                "accuracy": accuracy,
                "correct": accuracy > 0,
                **{key: value for key, value in meta.items() if key not in {"dataset", "question", "answer"}},
            })
    return results


def calculate_metrics(results: List[Dict]) -> Dict:
    if not results:
        return {"accuracy": 0.0}
    metrics = {"accuracy": sum(item["correct"] for item in results) / len(results)}
    for key in ("task", "problem_version", "subject"):
        if any(key in item for item in results):
            groups = {}
            for item in results:
                if key in item:
                    groups.setdefault(item[key], []).append(item["correct"])
            metrics["sub_accuracies"] = {name: sum(vals) / len(vals) for name, vals in groups.items()}
    return metrics
