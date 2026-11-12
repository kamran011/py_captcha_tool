"""
CSV import for the site benchmark.

Expected columns (case-insensitive): Company, Website, City, State, Category,
Platform, Captcha, Contact Page. Only `Contact Page` is required. Every row is
validated individually: bad rows are reported with their line number and
reason and never abort the import. Re-importing is idempotent (rows are matched
on Contact Page).

URL rules (the scanner fetches these, so they are checked here too): http/https
only, a real hostname, no embedded credentials, no whitespace or control
characters, at most 2048 characters.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from sqlalchemy.orm import Session

from app.models import Site
from app.recognition.provider_detector import normalise_label

MAX_BYTES = 2_000_000
MAX_ROWS = 5000
MAX_URL = 2048
KNOWN_LABELS = {"recaptcha", "recaptcha_invisible", "hcaptcha", "turnstile", "none"}
_FIELDS = {"company": "company", "website": "website", "city": "city", "state": "state",
           "category": "category", "platform": "platform", "captcha": "labelled_captcha",
           "contact page": "contact_page"}
_BAD_URL_CHARS = re.compile(r"[\s\x00-\x1f\x7f<>\"']")
_SPACE20 = re.compile(r"\s20(?=[A-Z])")        # "Sioux 20Falls": a lost % from "Sioux%20Falls"


@dataclass
class ImportReport:
    rows: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)


class CsvError(ValueError):
    pass


def validate_url(url: str) -> str | None:
    """Return a reason the URL must not be scanned, or None if it is acceptable."""
    if not url:
        return "empty_url"
    if len(url) > MAX_URL:
        return "url_too_long"
    if _BAD_URL_CHARS.search(url):
        return "invalid_characters"
    try:
        u = urlparse(url)
        port = u.port
    except ValueError:
        return "unparsable_url"
    if u.scheme not in ("http", "https"):
        return "scheme_not_http"
    if not u.hostname or "." not in u.hostname:
        return "no_hostname"
    if u.username or u.password:
        return "credentials_in_url"
    if port not in (None, 80, 443):
        return "unusual_port"
    return None


def _clean(v: str | None) -> str | None:
    v = re.sub(r"\s+", " ", (v or "")).strip()
    return v or None


def parse_csv(content: bytes | str) -> ImportReport:
    if isinstance(content, bytes):
        if len(content) > MAX_BYTES:
            raise CsvError(f"file too large (limit {MAX_BYTES} bytes)")
        content = content.decode("utf-8-sig", errors="replace")
    elif len(content) > MAX_BYTES:
        raise CsvError(f"file too large (limit {MAX_BYTES} bytes)")
    reader = csv.reader(io.StringIO(content))
    try:
        header = next(reader)
    except StopIteration:
        raise CsvError("empty file") from None
    index = {_FIELDS[h.strip().lower()]: i for i, h in enumerate(header) if h.strip().lower() in _FIELDS}
    if "contact_page" not in index:
        raise CsvError("missing required column: Contact Page")

    rep, seen = ImportReport(), set()
    for lineno, raw in enumerate(reader, start=2):
        if not any(c.strip() for c in raw):
            continue
        if len(rep.rows) + len(rep.errors) >= MAX_ROWS:
            raise CsvError(f"too many rows (limit {MAX_ROWS})")
        row = {k: _clean(raw[i]) if i < len(raw) else None for k, i in index.items()}
        url = (row.get("contact_page") or "").strip()
        problem = validate_url(url)
        if problem:
            rep.errors.append({"line": lineno, "reason": problem, "company": row.get("company")})
            continue
        if url in seen:
            rep.errors.append({"line": lineno, "reason": "duplicate_contact_page", "company": row.get("company")})
            continue
        seen.add(url)
        if row.get("city"):
            row["city"] = _SPACE20.sub(" ", row["city"])
        label = normalise_label(row.get("labelled_captcha"))
        if label not in KNOWN_LABELS:
            rep.warnings.append({"line": lineno, "reason": f"unknown_label:{label[:40]}", "company": row.get("company")})
        row["contact_page"] = url
        rep.rows.append(row)
    return rep


def import_sites(db: Session, rows: list[dict]) -> dict:
    """Upsert rows keyed on contact_page. Existing scan results are kept."""
    existing = {s.contact_page: s for s in db.query(Site).filter(Site.contact_page.in_([r["contact_page"] for r in rows]))} if rows else {}
    inserted = updated = unchanged = 0
    for r in rows:
        site = existing.get(r["contact_page"])
        if site is None:
            db.add(Site(**r))
            inserted += 1
            continue
        changed = False
        for k, v in r.items():
            if getattr(site, k) != v:
                setattr(site, k, v)
                changed = True
        updated += changed
        unchanged += not changed
    db.commit()
    return {"inserted": inserted, "updated": updated, "unchanged": unchanged}
