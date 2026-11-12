"""
Application configuration loaded exclusively from environment variables or .env.
Secrets use SecretStr so pydantic never logs their plain-text value.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API ──────────────────────────────────────────────────────────────────
    api_key: SecretStr = SecretStr("changeme-local")
    api_v1_prefix: str = "/api/v1"
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False

    # Abuse guards (in-process; one API process serves one reviewer or integrator)
    rate_limit_per_min: int = 120          # requests per minute per API key
    max_concurrent_solves: int = 4         # CPU-bound solves running at once
    solve_queue_timeout_s: float = 2.0     # wait for a free slot before answering 503
    max_upload_bytes: int = 5_000_000

    # ── Storage ──────────────────────────────────────────────────────────────
    database_url: str = "sqlite:///data/captcha_tool.db"

    # ── Solver ───────────────────────────────────────────────────────────────
    crnn_model_path: Path = Path("data/models/crnn.onnx")
    confidence_threshold: float = 0.60      # below this → try next solver
    solver_timeout_s: float = 10.0

    # ── Token verification ───────────────────────────────────────────────────
    # Official reCAPTCHA v2 test keys (always pass on Google's server)
    recaptcha_sitekey: str = "6LeIxAcTAAAAAJcZVRqyHh71UMIEGNQ_MXjiZKhI"
    recaptcha_secret: SecretStr = SecretStr(
        "6LeIxAcTAAAAAGG-vFI1TnRWxMZNFuojJ4WifJWe"
    )

    # Official hCaptcha test keys (always pass)
    hcaptcha_sitekey: str = "10000000-ffff-ffff-ffff-000000000001"
    hcaptcha_secret: SecretStr = SecretStr(
        "0x0000000000000000000000000000000000000000"
    )

    # Official Cloudflare Turnstile test keys
    # sitekey → "1x00..." = always pass, "2x00..." = always block,
    #            "3x00..." = force interactive challenge
    turnstile_sitekey: str = "1x00000000000000000000AA"
    turnstile_secret: SecretStr = SecretStr(
        "1x0000000000000000000000000000000AA"
    )

    recaptcha_verify_url: str = "https://www.google.com/recaptcha/api/siteverify"
    hcaptcha_verify_url: str = "https://api.hcaptcha.com/siteverify"
    turnstile_verify_url: str = "https://challenges.cloudflare.com/turnstile/v0/siteverify"

    verify_timeout_s: float = 5.0
    verify_retries: int = 1          # extra attempts, only where retrying is safe
    verify_backoff_s: float = 0.3
    # Ceiling on token age. Each provider also has its own lifetime (reCAPTCHA and
    # hCaptcha 120 s, Turnstile 300 s); the effective limit is the smallest of
    # the two and any per-request max_age_s.
    token_max_age_s: int = 300

    # ── Dashboard (Streamlit) ────────────────────────────────────────────────
    streamlit_host: str = "127.0.0.1"
    streamlit_port: int = 8501

    # ── Optional third-party vision solver (M7) ──────────────────────────────
    # Off unless VISION_API_KEY is set. When on it is the LAST resort in the text
    # pipeline, so an image is sent to the provider only after the local models
    # failed to give an accepted answer. Cost comes from the provider's reported
    # token usage times the prices below.
    vision_api_key: SecretStr | None = None
    vision_enabled: bool = True                 # master switch; has no effect without a key
    vision_provider: Literal["anthropic", "openai"] = "anthropic"
    vision_model: str = "claude-haiku-5-5"
    vision_base_url: str | None = None          # default per provider; set for an OpenAI-compatible endpoint
    vision_max_tokens: int = 200
    vision_token_param: str = "max_completion_tokens"   # OpenAI-style name; some compatible servers want max_tokens
    vision_timeout_s: float = 15.0
    vision_retries: int = 2
    # USD per million tokens; required for a model that is not in the built-in table
    vision_price_in_per_mtok: float | None = None
    vision_price_out_per_mtok: float | None = None
    # Hard spending guards (in-process; the daily total is re-read from the database at startup)
    vision_budget_usd_per_day: float = 1.00
    vision_max_calls_per_min: int = 20

    # ── Site scan ─────────────────────────────────────────────────────────────
    scan_concurrency: int = 5
    scan_timeout_s: float = 20.0
    scan_user_agent: str = (
        "CaptchaRecognitionTool/1.0 (take-home assignment; "
        "read-only detection scan; no form interaction)"
    )

    def secret_values(self) -> list[str]:
        """Return all secret values in plain text for the redacting log filter."""
        secrets = []
        for field_name in type(self).model_fields:
            val = getattr(self, field_name)
            if isinstance(val, SecretStr) and val.get_secret_value():
                secrets.append(val.get_secret_value())
        return [s for s in secrets if len(s) >= 8]  # ignore trivially short


settings = Settings()
