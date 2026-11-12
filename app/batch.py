"""
Batch evaluation: run the solve pipeline over many labelled images and report
accuracy, coverage, confidence and latency.

Datasets are NAMED folders under data/synthetic/ (each with a labels.csv of
`filename,text`); callers never pass a filesystem path, so the API cannot be
pointed at arbitrary server files.
"""
from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path

from app.solvers.metrics import char_accuracy, latency_summary
from app.solvers.pipeline import SolvePipeline

DATA_ROOT = Path("data/synthetic")
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
MAX_ITEMS = 1000
WORST = 15


class DatasetError(ValueError):
    pass


@dataclass
class BatchItem:
    name: str
    data: bytes
    expected: str | None = None


def list_datasets(root: Path | None = None) -> list[dict]:
    root = root or DATA_ROOT            # resolved per call, not bound at import
    out = []
    if root.exists():
        for d in sorted(p for p in root.iterdir() if p.is_dir() and NAME_RE.match(p.name)):
            labels = d / "labels.csv"
            if labels.exists():
                with open(labels, encoding="utf-8") as f:
                    out.append({"name": d.name, "count": max(sum(1 for _ in f) - 1, 0)})
    return out


def load_dataset(name: str, limit: int, root: Path | None = None) -> list[BatchItem]:
    root = root or DATA_ROOT
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise DatasetError("invalid dataset name")
    folder = (root / name).resolve()
    if folder.parent != root.resolve() or not (folder / "labels.csv").exists():
        raise DatasetError(f"unknown dataset {name!r}")
    items: list[BatchItem] = []
    with open(folder / "labels.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if len(items) >= max(1, min(limit, MAX_ITEMS)):
                break
            path = (folder / row["filename"]).resolve()
            if path.parent != folder or not path.is_file():      # no traversal via labels.csv
                continue
            items.append(BatchItem(row["filename"], path.read_bytes(), row["text"].strip().upper()))
    if not items:
        raise DatasetError(f"dataset {name!r} has no usable images")
    return items


def run_batch(pipeline: SolvePipeline, items: list[BatchItem], dataset: str = "upload",
              time_budget_s: float = 60.0, allow_external: bool = False) -> dict:
    """
    Solve every item sequentially and summarise. Stops early if the time budget runs out.
    External (paid) solvers are OFF unless asked for: a batch must never spend money by surprise.
    """
    t0 = time.perf_counter()
    rows: list[dict] = []
    truncated = False
    for it in items:
        if time.perf_counter() - t0 > time_budget_s:
            truncated = True
            break
        res = pipeline.solve(it.data, allow_external=allow_external)
        exp = it.expected
        rows.append({
            "name": it.name, "expected": exp, "predicted": res.text, "type": res.challenge_type,
            "accepted": res.accepted, "confidence": res.confidence, "ms": res.total_ms,
            "error": res.error, "method": res.method,
            "cost": res.cost_usd,
            "exact": (res.text == exp) if exp is not None else None,
            "char_acc": char_accuracy(res.text, exp) if exp is not None else None,
        })

    labelled = [r for r in rows if r["expected"] is not None]
    n, nl = len(rows), len(labelled)
    accepted = [r for r in labelled if r["accepted"]]

    def rate(xs: list[dict], key: str = "exact") -> float | None:
        return round(sum(1 for r in xs if r[key]) / len(xs), 4) if xs else None

    by_type: dict[str, dict] = {}
    for t in sorted({r["type"] for r in labelled}):
        grp = [r for r in labelled if r["type"] == t]
        by_type[t] = {"n": len(grp), "exact_match": rate(grp)}
    errors: dict[str, int] = {}
    for r in rows:
        if r["error"]:
            errors[r["error"].split(":")[0]] = errors.get(r["error"].split(":")[0], 0) + 1

    wrong = sorted((r for r in labelled if not r["exact"]), key=lambda r: -r["confidence"])
    return {
        "dataset": dataset,
        "n": n,
        "labelled": nl,
        "truncated": truncated,
        "total_seconds": round(time.perf_counter() - t0, 2),
        "exact_match": rate(labelled),
        "char_accuracy": round(sum(r["char_acc"] for r in labelled) / nl, 4) if nl else None,
        "mean_confidence": round(sum(r["confidence"] for r in rows) / n, 4) if n else None,
        "accepted": sum(1 for r in rows if r["accepted"]),
        "coverage": round(sum(1 for r in rows if r["accepted"]) / n, 4) if n else None,
        "precision_on_accepted": rate(accepted),
        "confident_wrong": sum(1 for r in accepted if not r["exact"]),
        "latency_ms": latency_summary([r["ms"] for r in rows]),
        "by_type": by_type,
        "by_error": errors,
        "worst": [{k: r[k] for k in ("name", "expected", "predicted", "confidence", "error")}
                  for r in wrong[:WORST]],
        "cost_usd": round(sum(r["cost"] for r in rows), 6),
        "external_allowed": allow_external,
        "results": rows,
    }
