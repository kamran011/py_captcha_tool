"""
Token verification endpoints.

POST /api/v1/verify      X-API-Key required. For integrators.
POST /demo-api/verify    No key, loopback clients only. Used by the local demo pages so
                         no API key is embedded in browser-visible HTML.

Both always answer with the structured VerifyResult, including for missing and
malformed tokens (never a bare framework 422).
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.api.limits import guard
from app.verification.service import VerificationService

router = APIRouter(tags=["verification"])
_LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}

_service: VerificationService | None = None


def get_service() -> VerificationService:
    global _service
    if _service is None:
        _service = VerificationService()
    return _service


class VerifyRequest(BaseModel):
    provider: str = Field(description="recaptcha | hcaptcha | turnstile")
    token: str | None = Field(default=None, description="Token from the client-side widget")
    remote_ip: str | None = None
    max_age_s: int | None = Field(default=None, ge=1, le=86_400,
                                  description="Stricter age limit for this call; never loosens the provider lifetime")
    test_scenario: Literal["pass", "fail", "spent"] | None = Field(
        default=None, description="Turnstile dummy-secret scenarios; only while the published dummy secret is configured")


def _run(req: VerifyRequest, db: Session, svc: VerificationService) -> JSONResponse:
    res = svc.verify(db, req.provider, req.token, remote_ip=req.remote_ip,
                     max_age_s=req.max_age_s, test_scenario=req.test_scenario)
    return JSONResponse(status_code=res.http_status, content=res.as_dict())


@router.post("/api/v1/verify", dependencies=[Depends(guard)])
def verify(req: VerifyRequest, db: Session = Depends(get_db),
           svc: VerificationService = Depends(get_service)):
    return _run(req, db, svc)


@router.post("/demo-api/verify", include_in_schema=False)
def verify_demo(req: VerifyRequest, request: Request, db: Session = Depends(get_db),
                svc: VerificationService = Depends(get_service)):
    if (request.client.host if request.client else "") not in _LOOPBACK:
        raise HTTPException(status_code=403, detail="demo endpoint is local-only")
    return _run(req, db, svc)
