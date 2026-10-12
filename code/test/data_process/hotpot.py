import json
from typing import Dict, List


def format_search_results(paragraphs: List[Dict], max_results: int = 5) -> str:
    entries = []
    for idx, item in enumerate(paragraphs[:max_results], 1):
        title = str(item.get("title", "")).replace("\n", " ").strip()
        context = str(item.get("context", "")).replace("\n", " ").strip()
        entries.append(f"[webpage {idx} begin]\nTitle: {title}\nContent: {context}\n[webpage {idx} end]")
    return "Retrieved information:\n" + "\n".join(entries) + "\nQusetion:\n"


def hotpotqa(data_path: str, max_results: int = 5) -> List[Dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    samples = []
    for item in dataset:
        contexts = []
        for idx, entry in enumerate(item.get("context", [])):
            if isinstance(entry, list) and len(entry) >= 2:
                title = str(entry[0])
                paragraphs = entry[1] if isinstance(entry[1], list) else []
            else:
                title, paragraphs = "Unknown", []
            contexts.append({"idx": idx, "title": title, "context": " ".join(map(str, paragraphs)).strip()})
        samples.append({
            "problem": item["question"],
            "passage": format_search_results(contexts, max_results),
            "answer": item["answer"],
            "document": [{"Title": doc["title"], "Content": doc["context"]} for doc in contexts[:max_results]],
        })
    return samples
