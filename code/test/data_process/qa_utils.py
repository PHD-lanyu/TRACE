import re
import string
from typing import Any, Iterable, Optional

ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def normalize_answer(text: Any) -> str:
    text = str(text)
    text = text.replace("_", " ")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = text.lower()
    text = "".join(ch if ch not in set(string.punctuation + "‘’´`") else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def bool_mapping(text: Any) -> str:
    if text == "True":
        return "yes"
    if text == "False":
        return "no"
    return str(text)


def extract_answer(text: str) -> Optional[str]:
    match = ANSWER_PATTERN.search(text or "")
    return match.group(1).strip() if match else None


def exact_match(prediction: str, ground_truth: Any) -> bool:
    truths: Iterable[Any] = [ground_truth] if isinstance(ground_truth, str) else ground_truth
    normalized_pred = normalize_answer(bool_mapping(prediction))
    return any(normalized_pred == normalize_answer(bool_mapping(truth)) for truth in truths)
