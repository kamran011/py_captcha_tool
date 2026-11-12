"""
CAPTCHA Solving & Verification Tool — FastAPI application entry point.

Two processes serve the full stack:
  uvicorn app.main:app --host 127.0.0.1 --port 8000
  streamlit run dashboard/streamlit_app.py

This module registers the lifespan (DB init, logging setup), mounts the
static demo pages for the three provider widgets, and includes all routers.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.batch import router as batch_router
from app.api.health import router as health_router
from app.api.limits import ApiError, error_response_body
from app.api.sites import router as sites_router
from app.api.solvers import router as solvers_router
from app.api.solve import router as solve_router
from app.api.stats import router as stats_router
from app.api.verify import router as verify_router
from app.config import settings
from app.db import init_db
from app.logging_config import configure_logging, register_secrets

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    configure_logging(level=logging.DEBUG if settings.debug else logging.INFO)
    register_secrets(settings.secret_values())
    logger.info("Starting CAPTCHA Tool API v1.0.0")
    init_db()
    from app.db import SessionLocal
    from app.solvers.vision_api_solver import availability, preload_spend

    if availability()[0]:      # external solver is on: a restart must not reset today's budget
        with SessionLocal() as db:
            logger.info("Vision solver enabled; spend already recorded today: $%.4f", preload_spend(db))
    else:
        logger.info("Vision solver off (%s)", availability()[1])
    if settings.api_key.get_secret_value() == "changeme-local":
        logger.warning("API_KEY is still the shipped default value: fine on localhost, "
                       "change it before exposing this service to any network")

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("CAPTCHA Tool API shutting down")


app = FastAPI(
    title="CAPTCHA Solving & Verification Tool",
    description=(
        "Recognises and solves synthetic CAPTCHA images, validates provider "
        "tokens server-side using official test keys, and benchmarks a "
        "provider-detection scan against a labelled site dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(stats_router)
app.include_router(solve_router)
app.include_router(batch_router)
app.include_router(sites_router)
app.include_router(solvers_router)
app.include_router(verify_router)


@app.exception_handler(ApiError)
async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(status_code=exc.status, content=error_response_body(exc), headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Structured 422 that never echoes the submitted values (a token may be among them)."""
    errors = [{"loc": [str(p) for p in e["loc"]], "msg": e["msg"], "type": e["type"]}
              for e in exc.errors()]
    return JSONResponse(status_code=422,
                        content={"ok": False, "reason": "invalid_request", "errors": errors})

# ── Static: provider demo pages (served at /demo/) ───────────────────────────
_demo_dir = Path(__file__).parent.parent / "web" / "demo"
_demo_dir.mkdir(parents=True, exist_ok=True)
app.mount("/demo", StaticFiles(directory=str(_demo_dir), html=True), name="demo")
