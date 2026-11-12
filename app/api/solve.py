"""
Challenge endpoints (all versioned under /api/v1, X-API-Key required, rate limited):

  POST /classify        image -> text | math | unsupported, plus quality metrics
  POST /solve           image -> answer, methods tried, cost; recorded as an attempt
  POST /detect          page HTML -> which CAPTCHA provider it embeds
  GET  /attempts        recent solve attempts (paged)
  GET  /verifications   recent token verifications (paged; fingerprints only)

Status codes for /solve: 200 answered (check `accepted`), 400 not an image / empty,
413 too large, 422 unsupported or unreadable challenge, 503 busy, 500 internal.
"""
from __future__ import annotations

import re
import threading
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.api.limits import ApiError, guard, read_upload, solve_slot
from app.models import Attempt, Verification
from app.recognition.image_classifier import classify_image
from app.recognition.provider_detector import MAX_HTML_BYTES, detect
from app.recognition.quality import assess
from app.solvers.base import load_gray
from app.solvers.pipeline import PipelineResult, SolvePipeline, record_attempt

router = APIRouter(prefix="/api/v1", tags=["challenges"], dependencies=[Depends(guard)])

_pipeline: SolvePipeline | None = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> SolvePipeline:
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = SolvePipeline()
        return _pipeline


_NAME_CLEAN = re.compile(r"[^\w.\- ]")


def _safe_name(name: str | None) -> str | None:
    """Keep only the base name, without path or control characters."""
    if not name:
        return None
    return _NAME_CLEAN.sub("_", name.replace("\\", "/").rsplit("/", 1)[-1])[:120] or None


def _status_for(res: PipelineResult) -> int:
    err = res.error or ""
    if err.startswith("invalid_image"):
        return 400
    if err == "unsupported_challenge" or err.startswith("poor_quality"):
        return 422
    if err == "internal_error":
        return 500
    return 200


def _decode(data: bytes):
    try:
        return load_gray(data)
    except ValueError as exc:
        raise ApiError(400, "invalid_image", str(exc)) from None


# ── models (OpenAPI) ─────────────────────────────────────────────────────────
class SolverAttempt(BaseModel):
    method: str
    text: str | None = None
    confidence: float = 0.0
    ms: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None
    accepted: bool = False


class SolveResponse(BaseModel):
    attempt_id: int | None = None
    challenge_type: str
    classifier_confidence: float
    text: str | None = Field(None, description="Answer (for math: the evaluated result)")
    confidence: float
    method: str | None
    accepted: bool = Field(description="True when a solver answered above its calibrated threshold")
    success: bool | None = Field(None, description="Equals the expected text when one was supplied, else `accepted`")
    error: str | None = None
    attempts: list[SolverAttempt]
    total_ms: float
    cost_usd: float
    detail: dict


class ClassifyResponse(BaseModel):
    type: Literal["text", "math", "unsupported"]
    confidence: float
    reasons: list[str]
    quality: dict


class DetectRequest(BaseModel):
    html: str = Field(description="Page HTML (static, or a browser-rendered DOM)")


# ── endpoints ────────────────────────────────────────────────────────────────
@router.post("/classify", response_model=ClassifyResponse)
def classify(file: UploadFile = File(..., description="Challenge image")):
    gray = _decode(read_upload(file))
    cls = classify_image(gray)
    q = assess(gray)
    return {"type": cls.type, "confidence": cls.confidence, "reasons": cls.reasons,
            "quality": {"ok": q.ok, "reason": q.reason, **q.metrics}}


@router.post("/solve", response_model=SolveResponse)
def solve(
    file: UploadFile = File(..., description="Challenge image (PNG/JPEG/...)"),
    expected: str | None = Form(None, max_length=64, description="Known answer, to record accuracy"),
    challenge_type: Literal["text", "math"] | None = Form(None, description="Skip classification"),
    allow_external: bool = Form(True, description="Allow the optional third-party vision solver (it receives the image). "
                                                    "Has no effect unless VISION_API_KEY is configured."),
    db: Session = Depends(get_db),
    pipeline: SolvePipeline = Depends(get_pipeline),
):
    data = read_upload(file)
    with solve_slot():
        res = pipeline.solve(data, challenge_type, allow_external=allow_external)
    status = _status_for(res)
    body = res.as_dict()
    if status == 400:
        # Not an image at all: never reached recognition, so it is not an attempt
        # (recording it would pollute the accuracy and volume statistics).
        body.update(attempt_id=None, success=False)
        return JSONResponse(status_code=400, content=jsonable_encoder(body))
    exp = expected.strip().upper() if expected and expected.strip() else None
    row = record_attempt(db, res, expected=exp, filename=_safe_name(file.filename))
    body.update(attempt_id=row.id, success=row.success)
    return JSONResponse(status_code=status, content=jsonable_encoder(body))


@router.post("/detect")
def detect_provider(req: DetectRequest):
    if len(req.html) > MAX_HTML_BYTES:
        raise ApiError(413, "html_too_large", f"limit is {MAX_HTML_BYTES} characters")
    return detect(req.html).as_dict()


@router.get("/attempts")
def attempts(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    challenge_type: Literal["text", "math", "unsupported"] | None = None,
    db: Session = Depends(get_db),
):
    q = db.query(Attempt)
    if challenge_type:
        q = q.filter(Attempt.challenge_type == challenge_type)
    total = q.with_entities(func.count(Attempt.id)).scalar() or 0
    rows = q.order_by(Attempt.id.desc()).offset(offset).limit(limit).all()
    return {"total": total, "limit": limit, "offset": offset, "items": [
        {"id": a.id, "created_at": a.created_at.isoformat() if a.created_at else None,
         "challenge_type": a.challenge_type, "method": a.method_used, "predicted": a.predicted_text,
         "expected": a.expected_text, "success": a.success, "confidence": a.confidence,
         "processing_ms": a.processing_ms, "cost_usd": a.cost_usd, "error": a.error,
         "filename": a.image_filename} for a in rows]}


@router.get("/verifications")
def verifications(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    total = db.query(func.count(Verification.id)).scalar() or 0
    rows = db.query(Verification).order_by(Verification.id.desc()).offset(offset).limit(limit).all()
    return {"total": total, "limit": limit, "offset": offset, "items": [
        {"id": v.id, "created_at": v.created_at.isoformat() if v.created_at else None,
         "provider": v.provider, "ok": v.ok, "reason": v.reason, "http_status": v.http_status,
         "latency_ms": v.latency_ms, "token_fingerprint": v.token_fingerprint} for v in rows]}
