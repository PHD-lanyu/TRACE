import json
from typing import Dict, List


def conflict(data_path: str, max_results: int = 5) -> List[Dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return [
        {
            "problem": item["question"],
            "passage": "Retrieved information:\n" + item["context"] + "\nQusetion:\n",
            "answer": item["answer"],
            "document": item["context"],
            "is_cf": item.get("is_cf"),
        }
        for item in dataset
    ]


def conflict_mix(data_path: str, max_results: int = 5) -> List[Dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return [
        {
            "problem": item["question"],
            "passage": "Retrieved information:\n" + item["mixcontext"] + "\nQusetion:\n",
            "answer": item["answer"],
            "document": item["mixcontext"],
            "is_cf": item.get("is_cf"),
        }
        for item in dataset
    ]
