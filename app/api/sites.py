"""
Site benchmark endpoints (X-API-Key, rate limited).

  POST /api/v1/sites/import         upload a CSV (or use the bundled data/sites.csv)
  GET  /api/v1/sites                imported sites with their scan results (paged, filterable)
  POST /api/v1/sites/scan           start a READ-ONLY detection scan in the background (202)
  GET  /api/v1/sites/scan           progress of the current / last scan
  POST /api/v1/sites/scan/cancel
  GET  /api/v1/sites/report         accuracy vs the CSV labels, confusion matrix

The scan only GETs each page and analyses the HTML; see app/sites/scanner.py for
exactly what it does and refuses to do.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import db as app_db
from app.api.deps import get_db
from app.api.limits import ApiError, guard, read_upload
from app.models import Site
from app.sites import scanner
from app.sites.importer import MAX_BYTES, CsvError, import_sites, parse_csv
from app.sites.report import build_report, detected_label

router = APIRouter(prefix="/api/v1/sites", tags=["sites"], dependencies=[Depends(guard)])
BUNDLED = Path("data/sites.csv")


def get_manager() -> scanner.ScanManager:
    return scanner.manager


def get_session_factory():
    return app_db.SessionLocal


def get_scan_runtime() -> dict:
    return {"cfg": scanner.ScanConfig.from_settings(), "client_factory": scanner.make_client,
            "resolver": scanner.default_resolver}


class ScanRequest(BaseModel):
    limit: int = Field(200, ge=1, le=500, description="Maximum pages to scan")
    rescan: bool = Field(False, description="Include pages that were already scanned")
    site_ids: list[int] | None = Field(None, max_length=500, description="Only these sites")


def _row(s: Site) -> dict:
    return {
        "id": s.id, "company": s.company, "platform": s.platform, "contact_page": s.contact_page,
        "label": s.label_normalised, "scan_status": s.scan_status,
        "detected": detected_label(s) if s.scan_status == "ok" else None,
        "variant": s.detected_variant, "sitekey": s.detected_sitekey,
        "confidence": s.detected_confidence, "needs_render": s.needs_render,
        "http_status": s.scan_http_status, "scan_ms": s.scan_ms, "final_url": s.final_url,
        "error": s.scan_error, "evidence": json.loads(s.detected_evidence) if s.detected_evidence else [],
    }


@router.post("/import")
def import_csv(
    file: UploadFile | None = File(None, description="CSV with a 'Contact Page' column"),
    use_bundled: bool = Form(False, description="Import the bundled data/sites.csv instead"),
    db: Session = Depends(get_db),
):
    if file is not None:
        data = read_upload(file, MAX_BYTES)
    elif use_bundled:
        if not BUNDLED.exists():
            raise ApiError(404, "no_bundled_csv", "data/sites.csv is not present")
        data = BUNDLED.read_bytes()
    else:
        raise ApiError(422, "no_input", "upload a CSV file or set use_bundled=true")
    try:
        report = parse_csv(data)
    except CsvError as exc:
        raise ApiError(422, "invalid_csv", str(exc)) from None
    result = import_sites(db, report.rows)
    return {**result, "rows_valid": len(report.rows), "rows_rejected": len(report.errors),
            "errors": report.errors[:50], "warnings": report.warnings[:50]}


@router.get("")
def list_sites(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    scan_status: str | None = Query(None, max_length=32),
    label: Literal["recaptcha", "recaptcha_invisible", "hcaptcha", "turnstile", "none"] | None = None,
    mismatch_only: bool = False,
    db: Session = Depends(get_db),
):
    rows = db.query(Site).order_by(Site.id).all()
    if scan_status == "unscanned":
        rows = [s for s in rows if s.scan_status is None]
    elif scan_status:
        rows = [s for s in rows if s.scan_status == scan_status]
    if label:
        rows = [s for s in rows if s.label_normalised == label]
    if mismatch_only:
        rows = [s for s in rows if s.scan_status == "ok" and detected_label(s) != s.label_normalised]
    return {"total": len(rows), "limit": limit, "offset": offset,
            "items": [_row(s) for s in rows[offset: offset + limit]]}


@router.post("/scan", status_code=202)
def start_scan(req: ScanRequest, db: Session = Depends(get_db),
               manager: scanner.ScanManager = Depends(get_manager),
               factory=Depends(get_session_factory), runtime: dict = Depends(get_scan_runtime)):
    q = db.query(Site.id, Site.contact_page).order_by(Site.id)
    if req.site_ids:
        q = q.filter(Site.id.in_(req.site_ids))
    if not req.rescan:
        q = q.filter(Site.scanned_at.is_(None))
    pairs = [(i, u) for i, u in q.limit(req.limit).all()]
    if not pairs:
        raise ApiError(422, "nothing_to_scan", "no matching unscanned sites; import a CSV or set rescan=true")
    try:
        job = manager.start(factory, pairs, **runtime)
    except scanner.ScanBusy:
        raise ApiError(409, "scan_running", "a scan is already running") from None
    return JSONResponse(status_code=202, content=job.as_dict())


@router.get("/scan")
def scan_status(manager: scanner.ScanManager = Depends(get_manager)):
    job = manager.current()
    return job.as_dict() if job else {"status": "idle"}


@router.post("/scan/cancel")
def cancel_scan(manager: scanner.ScanManager = Depends(get_manager)):
    return {"cancelled": manager.cancel()}


@router.get("/report")
def report(db: Session = Depends(get_db)):
    return build_report(db)
