"""Accuracy and latency metrics shared by the batch service and the evaluation scripts."""
from __future__ import annotations

import numpy as np


def levenshtein(a: str, b: str) -> int:
    """Edit distance (insert, delete, substitute), two-row dynamic programme."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_accuracy(pred: str | None, truth: str) -> float:
    """1 - edit_distance / len(truth), floored at 0. An empty prediction scores 0."""
    pred = pred or ""
    if not truth:
        return 1.0 if not pred else 0.0
    return max(0.0, 1.0 - levenshtein(pred, truth) / len(truth))


def latency_summary(values: list[float]) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"p50": None, "p95": None, "mean": None}
    return {"p50": round(float(np.percentile(vals, 50)), 2),
            "p95": round(float(np.percentile(vals, 95)), 2),
            "mean": round(float(np.mean(vals)), 2)}
