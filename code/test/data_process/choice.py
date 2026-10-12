import ast
from typing import Dict, List

import pandas as pd


def format_search_results(contexts: List[Dict], max_results: int = 5) -> str:
    entries = [
        f"[webpage {item['idx'] + 1} begin]\n{item['context']}\n[webpage {item['idx'] + 1} end]"
        for item in contexts[:max_results]
    ]
    return "Retrieved information:\n" + "\n".join(entries) + "\nQusetion:\n"


def choice(data_path: str, max_results: int = 5) -> List[Dict]:
    df = pd.read_excel(data_path)
    samples = []
    for _, row in df.iterrows():
        options = ast.literal_eval(row["option"]) if isinstance(row["option"], str) else row["option"]
        option_text = "\n".join(f"{chr(65 + idx)}. {option}" for idx, option in enumerate(options))
        contexts, documents = [], []
        for idx in range(1, 6):
            context = row.get(f"top{idx}", "")
            if pd.notna(context):
                contexts.append({"idx": idx - 1, "title": f"Result {idx}", "context": str(context)})
                documents.append(str(context))
        samples.append({
            "problem": f"{row['query']}\n{option_text} Your answer should only be A or B or C or D or E.",
            "passage": format_search_results(contexts, max_results),
            "answer": row["answer"],
            "document": documents,
        })
    return samples
