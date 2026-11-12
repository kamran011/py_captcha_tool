"""
Measure the whole solve pipeline on held-out synthetic data.

    python ml/evaluate_pipeline.py

Reports: classifier confusion, pipeline exact-match on the 1,000 text test
images, how often the fallback ran and whether it helped, how many wrong
answers were returned with high confidence, and math accuracy. Writes
data/models/pipeline_report.json.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.solvers.math_render import random_expression, render_expression  # noqa: E402
from app.solvers.pipeline import SolvePipeline  # noqa: E402
from app.solvers.segment_solver import SegmentSolver  # noqa: E402

TEST = Path("data/synthetic/test")


def main() -> None:
    rows = list(csv.DictReader(open(TEST / "labels.csv", encoding="utf-8")))
    pipe, seg = SolvePipeline(), SegmentSolver()
    cls, exact, fallback_used, fallback_fixed = Counter(), 0, 0, 0
    accepted_wrong, accepted, lat, seg_ok = 0, 0, [], 0
    for r in rows:
        img = cv2.imread(str(TEST / r["filename"]), cv2.IMREAD_GRAYSCALE)
        res = pipe.solve(img)
        cls[res.challenge_type] += 1
        lat.append(res.total_ms)
        hit = res.text == r["text"]
        exact += hit
        if res.accepted:
            accepted += 1
            accepted_wrong += not hit
        if len(res.attempts) > 1:
            fallback_used += 1
            fallback_fixed += hit
        seg_ok += seg.solve(img).text == r["text"]

    rng, math_ok, math_cls = random.Random(2024), 0, Counter()
    for _ in range(300):
        expr, answer = random_expression(rng)
        res = pipe.solve(render_expression(expr, rng))
        math_cls[res.challenge_type] += 1
        math_ok += res.text == str(answer)

    n = len(rows)
    report = {
        "text_images": n,
        "text_classified_as": dict(cls),
        "pipeline_exact_match": round(exact / n, 4),
        "pipeline_accepted": accepted,
        "accepted_but_wrong": accepted_wrong,
        "fallback_triggered": fallback_used,
        "fallback_answered_correctly": fallback_fixed,
        "segment_solver_alone_exact_match": round(seg_ok / n, 4),
        "latency_ms_p50": round(float(np.percentile(lat, 50)), 2),
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 2),
        "math_images": 300,
        "math_classified_as": dict(math_cls),
        "math_exact_match": round(math_ok / 300, 4),
    }
    print(json.dumps(report, indent=2))
    Path("data/models/pipeline_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
