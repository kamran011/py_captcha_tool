"""Thin HTTP client the Streamlit dashboard uses to talk to the FastAPI backend."""
from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from app.config import settings

_http = httpx.Client(timeout=30)


@dataclass
class ApiResult:
    status: int            # 0 = the API could not be reached
    data: dict

    @property
    def ok(self) -> bool:
        return self.status == 200

    @property
    def reachable(self) -> bool:
        return self.status != 0


class ApiClient:
    def __init__(self, base: str | None = None, key: str | None = None) -> None:
        self.base = (base or os.getenv("API_BASE") or f"http://127.0.0.1:{settings.port}").rstrip("/")
        self._key = key or os.getenv("API_KEY") or settings.api_key.get_secret_value()

    def _call(self, method: str, path: str, auth: bool = True, **kw) -> ApiResult:
        headers = {"X-API-Key": self._key} if auth else {}
        try:
            r = _http.request(method, self.base + path, headers=headers, **kw)
        except httpx.HTTPError as exc:
            return ApiResult(0, {"ok": False, "reason": "api_unreachable", "detail": type(exc).__name__})
        try:
            data = r.json()
        except ValueError:
            data = {"ok": False, "reason": "bad_response"}
        return ApiResult(r.status_code, data if isinstance(data, dict) else {"data": data})

    def health(self) -> ApiResult:
        return self._call("GET", "/health", auth=False)

    def stats(self) -> ApiResult:
        return self._call("GET", "/api/v1/stats")

    def attempts(self, limit: int = 20) -> ApiResult:
        return self._call("GET", "/api/v1/attempts", params={"limit": limit})

    def verifications(self, limit: int = 20) -> ApiResult:
        return self._call("GET", "/api/v1/verifications", params={"limit": limit})

    def solvers(self) -> ApiResult:
        return self._call("GET", "/api/v1/solvers")

    def solve(self, image: bytes, filename: str = "challenge.png", expected: str | None = None,
              challenge_type: str | None = None, allow_external: bool = True) -> ApiResult:
        form = {k: v for k, v in (("expected", expected), ("challenge_type", challenge_type)) if v}
        if not allow_external:
            form["allow_external"] = "false"          # keep the image on this machine
        return self._call("POST", "/api/v1/solve", files={"file": (filename, image)}, data=form)

    def classify(self, image: bytes, filename: str = "challenge.png") -> ApiResult:
        return self._call("POST", "/api/v1/classify", files={"file": (filename, image)})

    def detect(self, html: str) -> ApiResult:
        return self._call("POST", "/api/v1/detect", json={"html": html})

    def verify(self, provider: str, token: str, max_age_s: int | None = None,
               test_scenario: str | None = None) -> ApiResult:
        body: dict = {"provider": provider, "token": token}
        if max_age_s:
            body["max_age_s"] = max_age_s
        if test_scenario:
            body["test_scenario"] = test_scenario
        return self._call("POST", "/api/v1/verify", json=body)

    # ── batch ───────────────────────────────────────────────────────────────
    def batch_datasets(self) -> ApiResult:
        return self._call("GET", "/api/v1/batch/datasets")

    def batch_run(self, dataset: str, limit: int) -> ApiResult:
        return self._call("POST", "/api/v1/batch", json={"dataset": dataset, "limit": limit})

    def batch_upload(self, images: list[tuple[str, bytes]], labels: bytes | None = None) -> ApiResult:
        files = [("files", (name, data)) for name, data in images]
        if labels:
            files.append(("labels", ("labels.csv", labels)))
        return self._call("POST", "/api/v1/batch/upload", files=files)

    def batch_runs(self, limit: int = 20) -> ApiResult:
        return self._call("GET", "/api/v1/batch/runs", params={"limit": limit})

    # ── sites ───────────────────────────────────────────────────────────────
    def sites_import(self, csv_bytes: bytes | None = None) -> ApiResult:
        if csv_bytes is None:
            return self._call("POST", "/api/v1/sites/import", data={"use_bundled": "true"})
        return self._call("POST", "/api/v1/sites/import", files={"file": ("sites.csv", csv_bytes)})

    def sites_list(self, limit: int = 100, scan_status: str | None = None, label: str | None = None,
                   mismatch_only: bool = False) -> ApiResult:
        params = {"limit": limit, "mismatch_only": str(mismatch_only).lower()}
        if scan_status:
            params["scan_status"] = scan_status
        if label:
            params["label"] = label
        return self._call("GET", "/api/v1/sites", params=params)

    def sites_scan_start(self, limit: int = 200, rescan: bool = False) -> ApiResult:
        return self._call("POST", "/api/v1/sites/scan", json={"limit": limit, "rescan": rescan})

    def sites_scan_status(self) -> ApiResult:
        return self._call("GET", "/api/v1/sites/scan")

    def sites_scan_cancel(self) -> ApiResult:
        return self._call("POST", "/api/v1/sites/scan/cancel")

    def sites_report(self) -> ApiResult:
        return self._call("GET", "/api/v1/sites/report")
