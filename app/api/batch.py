"""
Batch accuracy endpoints (X-API-Key, rate limited; a batch holds one solver slot while it runs).

  GET  /api/v1/batch/datasets   named datasets available server-side
  POST /api/v1/batch            run a named dataset (<= 1000 images) and report accuracy
  POST /api/v1/batch/upload     run uploaded images (<= 100), optionally with a labels CSV
  GET  /api/v1/batch/runs       history of dataset runs
"""
from __future__ import annotations

import csv
import io
import json

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.api.limits import ApiError, guard, read_upload, solve_slot
from app.api.solve import _safe_name, get_pipeline
from app.batch import MAX_ITEMS, BatchItem, DatasetError, list_datasets, load_dataset, run_batch
from app.models import BatchRun
from app.solvers.pipeline import SolvePipeline

router = APIRouter(prefix="/api/v1/batch", tags=["batch"], dependencies=[Depends(guard)])

MAX_UPLOAD_FILES = 100
MAX_LABELS_BYTES = 1_000_000
TRAINING_NOTE = "used for training: accuracy on it is optimistic, evaluate on `test` or `math_test`"


class BatchRequest(BaseModel):
    dataset: str = Field(description="Name from GET /batch/datasets")
    limit: int = Field(200, ge=1, le=MAX_ITEMS)
    allow_external: bool = Field(False, description="Let the optional paid vision solver run (default off: a batch must not spend money by surprise)")


@router.get("/datasets")
def datasets():
    return {"datasets": [{**d, "note": TRAINING_NOTE if d["name"] == "train" else None}
                         for d in list_datasets()]}


def _store(db: Session, report: dict) -> int:
    summary = {k: v for k, v in report.items() if k != "results"}
    row = BatchRun(dataset=report["dataset"][:64], n=report["n"], exact_match=report["exact_match"],
                   char_accuracy=report["char_accuracy"], mean_confidence=report["mean_confidence"],
                   latency_p50_ms=report["latency_ms"]["p50"], latency_p95_ms=report["latency_ms"]["p95"],
                   report_json=json.dumps(summary))
    db.add(row)
    db.commit()
    return row.id


@router.post("")
def run_dataset(req: BatchRequest, db: Session = Depends(get_db),
                pipeline: SolvePipeline = Depends(get_pipeline)):
    try:
        items = load_dataset(req.dataset, req.limit)
    except DatasetError as exc:
        raise ApiError(404 if "unknown" in str(exc) else 422, "dataset_error", str(exc)) from None
    with solve_slot():
        report = run_batch(pipeline, items, dataset=req.dataset, allow_external=req.allow_external)
    report["run_id"] = _store(db, report)
    report.pop("results")                    # per-item rows are only returned for uploads
    return report


@router.post("/upload")
def run_upload(
    files: list[UploadFile] = File(..., description="Challenge images (max 100)"),
    labels: UploadFile | None = File(None, description="Optional CSV with columns filename,text"),
    allow_external: bool = Form(False, description="Let the optional paid vision solver run"),
    pipeline: SolvePipeline = Depends(get_pipeline),
):
    if len(files) > MAX_UPLOAD_FILES:
        raise ApiError(413, "too_many_files", f"limit is {MAX_UPLOAD_FILES} files")
    truth: dict[str, str] = {}
    if labels is not None:
        text = read_upload(labels, MAX_LABELS_BYTES).decode("utf-8-sig", errors="replace")
        for row in csv.DictReader(io.StringIO(text)):
            if row.get("filename") and row.get("text") is not None:
                truth[_safe_name(row["filename"]) or ""] = row["text"].strip().upper()
    items = []
    for f in files:
        name = _safe_name(f.filename) or "upload"
        items.append(BatchItem(name, read_upload(f), truth.get(name)))
    with solve_slot():
        return run_batch(pipeline, items, dataset="upload", allow_external=allow_external)


@router.get("/runs")
def runs(limit: int = 20, db: Session = Depends(get_db)):
    rows = db.query(BatchRun).order_by(BatchRun.id.desc()).limit(max(1, min(limit, 100))).all()
    return {"items": [{"id": r.id, "created_at": r.created_at.isoformat() if r.created_at else None,
                       "dataset": r.dataset, "n": r.n, "exact_match": r.exact_match,
                       "char_accuracy": r.char_accuracy, "mean_confidence": r.mean_confidence,
                       "latency_p50_ms": r.latency_p50_ms, "latency_p95_ms": r.latency_p95_ms}
                      for r in rows]}
