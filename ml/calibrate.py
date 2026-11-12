"""
Derive per-solver acceptance thresholds from measured precision.

    python ml/calibrate.py [--target 0.90] [--vision N]

For each solver, find the lowest confidence threshold at which answers accepted on
the VALIDATION split reach the target precision (with enough support). A solver
that never reaches it gets threshold null and becomes advisory: its answer is
reported but never accepted on its own. The test split is used only to report the
resulting precision and coverage.

`--vision N` also measures the optional third-party vision solver on the first N
validation images. THAT SPENDS REAL MONEY and sends those images to the provider,
so it only runs with an explicit N, a configured key, and the daily budget guard on;
the worst-case cost is printed first. Without it the vision solver stays advisory.

Output: data/models/calibration.json (read by app/solvers/pipeline.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402
from app.solvers.crnn_solver import CRNNSolver  # noqa: E402
from app.solvers.segment_solver import SegmentSolver  # noqa: E402

MIN_SUPPORT = 30
GRID = [round(x, 3) for x in np.arange(0.0, 1.0001, 0.01)]


def stats(conf: np.ndarray, hit: np.ndarray, t: float) -> dict:
    sel = conf >= t
    return {
        "coverage": round(float(sel.mean()), 4),
        "precision": round(float(hit[sel].mean()), 4) if sel.any() else None,
        "accepted": int(sel.sum()),
    }


def choose_threshold(conf: np.ndarray, hit: np.ndarray, target: float,
                     min_support: int = MIN_SUPPORT, floor: float = 0.0) -> float | None:
    """Lowest grid threshold whose accepted set reaches `target` precision with enough support.

    Returns None when no threshold does (the solver stays advisory), else max(threshold, floor).
    """
    for t in GRID:
        s = stats(conf, hit, t)
        if s["accepted"] >= min_support and s["precision"] is not None and s["precision"] >= target:
            return max(t, floor)
    return None


def collect(solver, split: str, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    folder = Path("data/synthetic") / split
    rows = list(csv.DictReader(open(folder / "labels.csv", encoding="utf-8")))[:limit]
    conf, hit = [], []
    for r in rows:
        gray = cv2.imread(str(folder / r["filename"]), cv2.IMREAD_GRAYSCALE)
        try:
            res = solver.solve(gray)
            conf.append(res.confidence)
            hit.append(res.text == r["text"])
        except Exception as exc:  # noqa: BLE001 - an unusable reply counts as a miss at zero confidence
            print(f"  {solver.name}: {r['filename']}: {type(exc).__name__}: {exc}", file=sys.stderr)
            conf.append(0.0)
            hit.append(False)
    return np.array(conf), np.array(hit, dtype=bool)


def entry(solver, vc, vh, tc, th, target) -> dict:
    t = choose_threshold(vc, vh, target, floor=settings.confidence_threshold)
    return {
        "threshold": t,
        "exact_match_val": round(float(vh.mean()), 4),
        "exact_match_test": round(float(th.mean()), 4) if th is not None else None,
        "val_at_threshold": stats(vc, vh, t) if t is not None else None,
        "test_at_threshold": stats(tc, th, t) if t is not None and th is not None else None,
        "advisory_only": t is None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.90)
    ap.add_argument("--vision", type=int, default=0, metavar="N",
                    help="also measure the paid vision solver on N validation images (spends money)")
    args = ap.parse_args()

    out = {"target_precision": args.target, "min_support": MIN_SUPPORT, "solvers": {}}
    for solver in (CRNNSolver(), SegmentSolver()):
        vc, vh = collect(solver, "val")
        tc, th = collect(solver, "test")
        out["solvers"][solver.name] = entry(solver, vc, vh, tc, th, args.target)

    if args.vision:
        from app.solvers.vision_api_solver import EST_INPUT_TOKENS, VisionApiSolver, availability, prices_for

        ok, why = availability()
        if not ok:
            sys.exit(f"cannot calibrate the vision solver: {why}")
        pin, pout = prices_for(settings)
        worst = args.vision * (EST_INPUT_TOKENS * pin + settings.vision_max_tokens * pout) / 1e6
        print(f"vision calibration: {args.vision} calls, worst case ${worst:.4f} "
              f"(model {settings.vision_model}); budget guard ${settings.vision_budget_usd_per_day:.2f}/day")
        solver = VisionApiSolver()
        vc, vh = collect(solver, "val", args.vision)
        out["solvers"][solver.name] = entry(solver, vc, vh, None, None, args.target)
        out["solvers"][solver.name]["images"] = args.vision
        out["solvers"][solver.name]["model"] = settings.vision_model
        out["solvers"][solver.name]["spent_usd"] = solver.budget.status()["spent_today_usd"]
    else:
        previous = {}
        try:
            previous = json.loads(Path("data/models/calibration.json").read_text(encoding="utf-8"))["solvers"]
        except (OSError, ValueError, KeyError):
            pass
        if "vision_api" in previous:                  # keep an earlier measurement
            out["solvers"]["vision_api"] = previous["vision_api"]

    print(json.dumps(out, indent=2))
    Path("data/models/calibration.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
