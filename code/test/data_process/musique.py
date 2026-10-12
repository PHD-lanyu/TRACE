import json
from typing import Dict, List


def format_search_results(paragraphs: List[Dict], max_results: int = 5) -> str:
    entries = []
    for idx, item in enumerate(paragraphs[:max_results], 1):
        entries.append(
            f"[webpage {idx} begin]\n"
            f"Title: {item.get('title', '')}\n"
            f"Content: {item.get('paragraph_text', '')}\n"
            f"[webpage {idx} end]"
        )
    return "Retrieved information:\n" + "\n".join(entries) + "\nQusetion:\n"


def musique(data_path: str, max_results: int = 5) -> List[Dict]:
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if all(key in item for key in ("question", "answer", "paragraphs")):
                samples.append({
                    "problem": item["question"],
                    "passage": format_search_results(item["paragraphs"], max_results),
                    "answer": item["answer"],
                    "document": item["paragraphs"][:max_results],
                })
    return samples
