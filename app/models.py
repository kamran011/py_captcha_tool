"""
SQLAlchemy ORM models.

Tables
------
attempts        One solve attempt (image → predicted text).
verifications   One provider token verification.
token_ledger    SHA-256 of every token seen; enables reuse and expiry checks.
sites           One CSV row + its detection-scan result.
"""
from __future__ import annotations

import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


class Base(DeclarativeBase):
    pass


class Attempt(Base):
    """One CAPTCHA solve attempt."""

    __tablename__ = "attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    # Input
    challenge_type: Mapped[str | None] = mapped_column(
        String(32)
    )  # text | math | unsupported
    image_filename: Mapped[str | None] = mapped_column(String(255))

    # Pipeline
    methods_tried: Mapped[str | None] = mapped_column(
        Text
    )  # JSON list of method names
    method_used: Mapped[str | None] = mapped_column(String(64))

    # Result
    predicted_text: Mapped[str | None] = mapped_column(Text)
    expected_text: Mapped[str | None] = mapped_column(Text)
    success: Mapped[bool | None] = mapped_column(Boolean)
    confidence: Mapped[float | None] = mapped_column(Float)
    processing_ms: Mapped[float | None] = mapped_column(Float)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text)


class Verification(Base):
    """One provider token verification call."""

    __tablename__ = "verifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    provider: Mapped[str | None] = mapped_column(
        String(32)
    )  # recaptcha | hcaptcha | turnstile
    variant: Mapped[str | None] = mapped_column(
        String(32)
    )  # v2_checkbox | v2_invisible | v3 | hcaptcha | turnstile

    ok: Mapped[bool | None] = mapped_column(Boolean)
    reason: Mapped[str | None] = mapped_column(
        String(128)
    )  # provider error code or our label
    http_status: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[float | None] = mapped_column(Float)

    # Token is never stored; only its fingerprint (first 12 hex of SHA-256)
    token_fingerprint: Mapped[str | None] = mapped_column(String(16))
    error: Mapped[str | None] = mapped_column(Text)


class TokenLedger(Base):
    """
    SHA-256 hash of every token this tool has verified.
    Used to detect reuse and local expiry without storing the token itself.
    """

    __tablename__ = "token_ledger"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    token_hash: Mapped[str] = mapped_column(
        String(64), unique=True, index=True
    )  # full SHA-256 hex
    token_fingerprint: Mapped[str] = mapped_column(
        String(16)
    )  # first 12 hex chars for logs
    provider: Mapped[str | None] = mapped_column(String(32))
    first_seen: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    # pending       claimed, provider verdict not yet recorded (or caller crashed)
    # consumed      provider gave a verdict: the token is spent
    # indeterminate provider call failed after the request may have been processed;
    #               one re-attempt is allowed (compare-and-swap back to pending)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    verdict: Mapped[str | None] = mapped_column(String(64))
    challenge_ts: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )


class Site(Base):
    """One row from the imported CSV plus the detection-scan result."""

    __tablename__ = "sites"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # CSV columns
    company: Mapped[str | None] = mapped_column(Text)
    website: Mapped[str | None] = mapped_column(Text)
    city: Mapped[str | None] = mapped_column(Text)
    state: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(Text)
    platform: Mapped[str | None] = mapped_column(Text)
    contact_page: Mapped[str] = mapped_column(String(2048), unique=True, index=True)
    labelled_captcha: Mapped[str | None] = mapped_column(
        Text
    )  # CSV "Captcha" column value

    # Scan result
    scanned_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    detected_provider: Mapped[str | None] = mapped_column(Text)
    detected_variant: Mapped[str | None] = mapped_column(Text)
    detected_sitekey: Mapped[str | None] = mapped_column(Text)
    detected_evidence: Mapped[str | None] = mapped_column(Text)  # JSON
    scan_status: Mapped[str | None] = mapped_column(
        String(32)
    )  # ok | error | timeout | skipped
    scan_error: Mapped[str | None] = mapped_column(Text)
    scan_http_status: Mapped[int | None] = mapped_column(Integer)
    final_url: Mapped[str | None] = mapped_column(Text)         # after redirects
    scan_ms: Mapped[float | None] = mapped_column(Float)
    detected_confidence: Mapped[float | None] = mapped_column(Float)
    # static HTML mentioned a provider but showed no widget: probably injected by
    # JavaScript, which this read-only scan deliberately does not execute
    needs_render: Mapped[bool | None] = mapped_column(Boolean)

    @property
    def label_normalised(self) -> str:
        """The CSV label in the detector's vocabulary (recaptcha | recaptcha_invisible | ...)."""
        from app.recognition.provider_detector import normalise_label

        return normalise_label(self.labelled_captcha)


class BatchRun(Base):
    """One batch accuracy run over a dataset."""

    __tablename__ = "batch_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    dataset: Mapped[str] = mapped_column(String(64))
    n: Mapped[int] = mapped_column(Integer)
    exact_match: Mapped[float | None] = mapped_column(Float)
    char_accuracy: Mapped[float | None] = mapped_column(Float)
    mean_confidence: Mapped[float | None] = mapped_column(Float)
    latency_p50_ms: Mapped[float | None] = mapped_column(Float)
    latency_p95_ms: Mapped[float | None] = mapped_column(Float)
    report_json: Mapped[str] = mapped_column(Text)
