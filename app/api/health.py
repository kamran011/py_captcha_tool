"""
GET /health — open endpoint, no API key required.
Returns service status and DB connectivity.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import get_db

router = APIRouter()

_VERSION = "1.0.0"


@router.get("/health", tags=["meta"])
def health(db: Session = Depends(get_db)):
    db_ok = False
    db_error: str | None = None
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception as exc:  # noqa: BLE001
        db_error = str(exc)

    return {
        "status": "ok" if db_ok else "degraded",
        "version": _VERSION,
        "db": {"ok": db_ok, "error": db_error},
    }
