"""Reward function for OCR-style reinforcement learning tasks."""

from difflib import SequenceMatcher
from typing import Any


def _normalized_levenshtein_similarity(prediction: str, target: str) -> float:
    """Return a soft reward between 0 and 1 based on edit similarity."""

    matcher = SequenceMatcher(None, prediction.lower().strip(), target.lower().strip())
    return matcher.ratio()


def compute_score(reward_input: dict[str, Any], format_weight: float = 0.2) -> dict[str, float]:
    """Compute OCR reward combining format sanity check and transcription accuracy."""

    if not isinstance(reward_input, dict):
        raise ValueError("Please provide a mapping with `response` and `ground_truth` keys.")

    response = str(reward_input.get("response", "")).strip()
    ground_truth = str(reward_input.get("ground_truth", "")).strip()

    format_score = 1.0 if "<answer>" not in response and "<think>" not in response else 0.0
    accuracy_score = _normalized_levenshtein_similarity(response, ground_truth) if ground_truth else 0.0

    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }
