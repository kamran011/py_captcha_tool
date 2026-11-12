"""
GET /api/v1/stats - everything the dashboard's Overview shows.

Keeps the original flat keys (attempts.total ... sites.scanned) and adds
breakdowns by challenge type and method, latency percentiles, cost, and
verification outcomes. Percentiles are computed over the most recent
PERCENTILE_WINDOW rows because SQLite has no percentile aggregate.
"""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends
from sqlalchemy import case, func
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.api.limits import guard
from app.models import Attempt, Site, Verification

router = APIRouter(prefix="/api/v1", tags=["stats"], dependencies=[Depends(guard)])
PERCENTILE_WINDOW = 5000


def _pcts(values: list[float]) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"p50": None, "p95": None}
    return {"p50": round(float(np.percentile(vals, 50)), 2), "p95": round(float(np.percentile(vals, 95)), 2)}


def _group(db: Session, model, key_col, ok_col) -> list[dict]:
    rows = (db.query(key_col, func.count(model.id), func.sum(case((ok_col == True, 1), else_=0)))  # noqa: E712
            .group_by(key_col).order_by(func.count(model.id).desc()).all())
    return [{"key": k or "none", "total": n, "successful": int(ok or 0)} for k, n, ok in rows]


@router.get("/stats")
def stats(db: Session = Depends(get_db)):
    total = db.query(func.count(Attempt.id)).scalar() or 0
    ok = db.query(func.count(Attempt.id)).filter(Attempt.success == True).scalar() or 0  # noqa: E712
    with_label = db.query(func.count(Attempt.id)).filter(Attempt.expected_text.isnot(None)).scalar() or 0
    label_ok = (db.query(func.count(Attempt.id)).filter(Attempt.expected_text.isnot(None))
                .filter(Attempt.success == True).scalar() or 0)  # noqa: E712
    avg_conf = db.query(func.avg(Attempt.confidence)).filter(Attempt.confidence.isnot(None)).scalar()
    cost = db.query(func.sum(Attempt.cost_usd)).scalar() or 0.0
    latency = [r[0] for r in db.query(Attempt.processing_ms).order_by(Attempt.id.desc()).limit(PERCENTILE_WINDOW)]

    v_total = db.query(func.count(Verification.id)).scalar() or 0
    v_ok = db.query(func.count(Verification.id)).filter(Verification.ok == True).scalar() or 0  # noqa: E712
    v_latency = [r[0] for r in db.query(Verification.latency_ms).order_by(Verification.id.desc()).limit(PERCENTILE_WINDOW)]
    reasons = (db.query(Verification.reason, func.count(Verification.id))
               .group_by(Verification.reason).order_by(func.count(Verification.id).desc()).all())

    return {
        "attempts": {
            "total": total, "successful": ok, "failed": total - ok,
            "success_rate": round(ok / total, 4) if total else None,
            "labelled": with_label,
            "labelled_accuracy": round(label_ok / with_label, 4) if with_label else None,
            "avg_confidence": round(avg_conf, 4) if avg_conf else None,
            "avg_processing_ms": round(float(np.mean([x for x in latency if x is not None])), 2)
                                 if any(x is not None for x in latency) else None,
            "latency_ms": _pcts(latency),
            "total_cost_usd": round(cost, 6),
            "avg_cost_usd": round(cost / total, 6) if total else 0.0,
            "by_type": _group(db, Attempt, Attempt.challenge_type, Attempt.success),
            "by_method": _group(db, Attempt, Attempt.method_used, Attempt.success),
            "cost_by_method": [
                {"key": k or "none", "calls": n, "cost_usd": round(float(c or 0.0), 6)}
                for k, n, c in db.query(Attempt.method_used, func.count(Attempt.id), func.sum(Attempt.cost_usd))
                .group_by(Attempt.method_used).order_by(func.sum(Attempt.cost_usd).desc()).all()],
        },
        "verifications": {
            "total": v_total, "successful": v_ok, "failed": v_total - v_ok,
            "latency_ms": _pcts(v_latency),
            "by_reason": [{"key": r or "none", "total": n} for r, n in reasons],
            "by_provider": _group(db, Verification, Verification.provider, Verification.ok),
        },
        "sites": {
            "total": db.query(func.count(Site.id)).scalar() or 0,
            "scanned": db.query(func.count(Site.id)).filter(Site.scan_status == "ok").scalar() or 0,
        },
    }
