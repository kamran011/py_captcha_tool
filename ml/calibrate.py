"""
Derive per-solver acceptance thresholds from measured precision.

    python ml/calibrate.py [--target 0.90] [--vision N]

For each solver, find the lowest confidence threshold at which answers accepted on
the VALIDATION split reach the target precision (with enough support). A solver
that never reaches it gets threshold null and becomes advisory: its answer is
reported but never accepted on its own. The test split is used only to report the
resulting precision and coverage.

`--vision N` also measures the optional third-party vision solver on N images. THAT
SPENDS REAL MONEY and sends those images to the provider, so it only runs with an
explicit N, a configured key, and the daily budget guard on (preloaded with the spend
already recorded today); the worst-case cost is printed first. Without it the vision
solver stays advisory.

The vision solver is the last tier: it only ever sees images the local models failed.
So it is measured on that population, not on the easy validation split: hard-sample
images from every difficulty tier (indices the dashboard never shows) that the local
chain does not accept. Alternate images form a validation half (threshold chosen) and
a test half (precision reported). The per-minute call cap is waited out, never scored.

Output: data/models/calibration.json (read by app/solvers/pipeline.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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


VISION_FIRST_INDEX = 100          # dashboard hard samples use indices 0-19; never calibrate on them


def vision_population(n: int) -> list[tuple[np.ndarray, str, str]]:
    """N hard images (round-robin over the tiers) that the local solvers do NOT accept."""
    from app.solvers.pipeline import SolvePipeline
    from dashboard.samples import HARD_LEVELS, hard_sample

    local = SolvePipeline()
    out, index, levels = [], VISION_FIRST_INDEX, list(HARD_LEVELS)
    while len(out) < n and index < VISION_FIRST_INDEX + 50 * n:
        for level in levels:
            png, text = hard_sample(level, index)
            gray = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
            res = local.solve(gray, challenge_type="text", allow_external=False)
            if not res.accepted and not (res.error or "").startswith("poor_quality"):
                out.append((gray, text, level))
            if len(out) >= n:
                break
        index += 1
    return out


def collect_vision(solver, items) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    from app.solvers.vision_api_solver import BudgetExceeded, RateLimited

    conf, hit, rows = [], [], []
    for i, (gray, text, level) in enumerate(items, 1):
        while True:
            try:
                res = solver.solve(gray)
                c, h, got, err = res.confidence, res.text == text, res.text, None
            except RateLimited:
                time.sleep(5)                 # the per-minute cap is a pause, not a wrong answer
                continue
            except BudgetExceeded as exc:
                sys.exit(f"stopped after {i - 1} images: {exc}")
            except Exception as exc:  # noqa: BLE001 - an unusable paid reply is a miss at zero confidence
                c, h, got, err = 0.0, False, None, f"{type(exc).__name__}: {exc}"
            break
        conf.append(c)
        hit.append(h)
        rows.append({"level": level, "hit": h})
        print(f"  [{i}/{len(items)}] {level:<9} truth={text:<6} read={got or '-':<6} conf={c:.2f} "
              f"{'OK' if h else 'miss'}{'  ' + err if err else ''}", file=sys.stderr)
    return np.array(conf), np.array(hit, dtype=bool), rows


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

        from app.db import SessionLocal
        from app.solvers.vision_api_solver import preload_spend

        ok, why = availability()
        if not ok:
            sys.exit(f"cannot calibrate the vision solver: {why}")
        pin, pout = prices_for(settings)
        solver = VisionApiSolver()
        with SessionLocal() as db:
            already = preload_spend(db, solver.budget)
        # 10K input tokens bounds every vision model measured here (gpt-4o-mini high detail ~8.7K)
        worst = args.vision * (max(EST_INPUT_TOKENS, 10_000) * pin + settings.vision_max_tokens * pout) / 1e6
        print(f"vision calibration: {args.vision} calls, worst case ${worst:.4f} (model {settings.vision_model}); "
              f"${already:.4f} already spent today of ${settings.vision_budget_usd_per_day:.2f}", file=sys.stderr)
        items = vision_population(args.vision)
        print(f"  {len(items)} hard images the local chain does not accept", file=sys.stderr)
        before = solver.budget.status()["spent_today_usd"]
        conf, hit, rows = collect_vision(solver, items)
        vc, vh, tc, th = conf[0::2], hit[0::2], conf[1::2], hit[1::2]
        e = entry(solver, vc, vh, tc, th, args.target)
        e.update(source="hard samples not accepted by the local chain", images=len(items),
                 model=settings.vision_model, provider=settings.vision_provider,
                 spent_usd=round(solver.budget.status()["spent_today_usd"] - before, 6),
                 exact_match_by_level={lv: round(float(np.mean([r["hit"] for r in rows if r["level"] == lv])), 3)
                                       for lv in dict.fromkeys(r["level"] for r in rows)})
        out["solvers"][solver.name] = e
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
