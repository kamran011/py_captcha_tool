"""
Evaluate the exported ONNX model on the held-out test split.

    python ml/evaluate.py [--split test]

Prints exact-match, character accuracy, mean confidence, p50/p95 latency and
writes data/models/eval_report.json.  The README quotes this output verbatim.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.common import ctc_greedy_decode, preprocess  # noqa: E402

MODEL = Path("data/models/crnn.onnx")


def char_accuracy(pred: str, truth: str) -> float:
    if not truth:
        return 0.0
    return SequenceMatcher(None, pred, truth).ratio()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    folder = Path("data/synthetic") / args.split
    with open(folder / "labels.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    exact, chars, confs, lat = [], [], [], []
    wrong = []
    for r in rows:
        img = cv2.imread(str(folder / r["filename"]), cv2.IMREAD_GRAYSCALE)
        x = preprocess(img)[None]
        t0 = time.perf_counter()
        logits = sess.run(None, {"image": x})[0][0]
        text, conf = ctc_greedy_decode(logits)
        lat.append((time.perf_counter() - t0) * 1000)
        exact.append(text == r["text"])
        chars.append(char_accuracy(text, r["text"]))
        confs.append(conf)
        if text != r["text"] and len(wrong) < 10:
            wrong.append({"file": r["filename"], "truth": r["text"], "pred": text,
                          "conf": round(conf, 3)})

    report = {
        "split": args.split,
        "n": len(rows),
        "exact_match": round(float(np.mean(exact)), 4),
        "char_accuracy": round(float(np.mean(chars)), 4),
        "mean_confidence": round(float(np.mean(confs)), 4),
        "latency_ms_p50": round(float(np.percentile(lat, 50)), 2),
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 2),
        "sample_errors": wrong,
    }
    print(json.dumps(report, indent=2))
    Path("data/models/eval_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
