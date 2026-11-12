"""
Solve pipeline: classify -> try solvers in order -> accept on confidence.

Every solver tried is recorded (method, text, confidence, latency, error), so
the response always shows what happened. A solver that errors, times out, is
unavailable, returns low confidence, or returns a length outside the expected
range falls through to the next one. Unsupported challenges are never sent to
a solver.
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Attempt
from app.logging_config import redact
from app.recognition.image_classifier import classify_image
from app.recognition.quality import assess
from app.solvers.base import Solver, SolverUnavailable, load_gray
from app.solvers.crnn_solver import CRNNSolver
from app.solvers.math_solver import MathSolver
from app.solvers.segment_solver import SegmentSolver
from app.solvers.vision_api_solver import VisionApiSolver
from app.solvers.vision_api_solver import availability as vision_availability
from ml.common import MAX_LEN, MIN_LEN

logger = logging.getLogger(__name__)
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="solver")


@dataclass
class PipelineResult:
    challenge_type: str
    classifier_confidence: float
    text: str | None = None
    confidence: float = 0.0
    method: str | None = None
    accepted: bool = False
    error: str | None = None
    attempts: list[dict] = field(default_factory=list)
    total_ms: float = 0.0
    cost_usd: float = 0.0
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def default_solvers(challenge_type: str) -> list[Solver]:
    if challenge_type == "math":
        return [MathSolver()]
    solvers: list[Solver] = [CRNNSolver(), SegmentSolver()]
    if vision_availability()[0]:            # last resort, only when a key (and a price) is configured
        solvers.append(VisionApiSolver())
    return solvers


CALIBRATION_PATH = Path("data/models/calibration.json")


def load_calibration(path: Path = CALIBRATION_PATH) -> dict:
    """Per-solver acceptance thresholds written by ml/calibrate.py ({} if absent)."""
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("solvers", {})
    except (OSError, ValueError):
        return {}


class SolvePipeline:
    """
    `calibration` maps solver name -> {"threshold": float | None}. A null
    threshold marks the solver advisory: it runs and is reported, but its answer
    is never accepted on its own (measured precision never reached the target).
    Solvers absent from the map use the default threshold, except a solver that sets
    `requires_calibration` (its confidence is a self-report, e.g. the external vision
    solver): without a measured entry it is advisory too.
    """

    def __init__(
        self,
        solvers_for=default_solvers,
        threshold: float | None = None,
        timeout_s: float | None = None,
        calibration: dict | None = None,
    ) -> None:
        self._solvers_for = solvers_for
        self.threshold = settings.confidence_threshold if threshold is None else threshold
        self.timeout_s = settings.solver_timeout_s if timeout_s is None else timeout_s
        self.calibration = load_calibration() if calibration is None else calibration
        self._cache: dict[str, list[Solver]] = {}

    def _threshold_for(self, solver: Solver) -> float | None:
        entry = self.calibration.get(solver.name)
        if entry and getattr(solver, "external", False) and entry.get("model") not in (None, settings.vision_model):
            entry = None            # measured on a different model: that threshold says nothing about this one
        if entry is None:
            return None if getattr(solver, "requires_calibration", False) else self.threshold
        return entry.get("threshold")

    def solvers_status(self, challenge_type: str = "text") -> list[dict]:
        """Name, role and acceptance threshold of each solver for a challenge type (for the API)."""
        out = []
        for s in self._solvers(challenge_type):
            limit = self._threshold_for(s)
            out.append({"name": s.name, "kind": "external" if getattr(s, "external", False) else "local",
                        "challenge_types": [challenge_type], "enabled": True,
                        "threshold": limit, "advisory_only": limit is None})
        return out

    def _solvers(self, challenge_type: str) -> list[Solver]:
        # Reuse instances so models and glyph tables load once
        if challenge_type not in self._cache:
            self._cache[challenge_type] = self._solvers_for(challenge_type)
        return self._cache[challenge_type]

    @staticmethod
    def _implausible(challenge_type: str, text: str) -> str | None:
        """Reason a confident answer is still not credible, else None."""
        if challenge_type == "text":
            if not MIN_LEN <= len(text) <= MAX_LEN:
                return "implausible_length"
            # The CRNN collapses unreadable input to one repeated letter ("MMMMM")
            if len(set(text)) == 1:
                return "degenerate_output"
            return None
        return None if text else "implausible_length"

    def solve(self, source: bytes | np.ndarray, challenge_type: str | None = None,
              allow_external: bool = True) -> PipelineResult:
        """
        Never raises: any unexpected failure becomes error="internal_error".
        `allow_external=False` keeps the image on this machine: solvers that send it to a
        third party (the vision API) are skipped for this request.
        """
        try:
            return self._solve(source, challenge_type, allow_external)
        except Exception:  # noqa: BLE001
            logger.exception("unexpected pipeline failure")
            return PipelineResult("unsupported", 0.0, error="internal_error")

    def _solve(self, source: bytes | np.ndarray, challenge_type: str | None,
               allow_external: bool = True) -> PipelineResult:
        t0 = time.perf_counter()
        try:
            gray = load_gray(source)
        except ValueError as exc:   # fixed, non-revealing messages from load_gray
            return PipelineResult("unsupported", 0.0, error=f"invalid_image: {exc}")

        if challenge_type in ("text", "math"):
            ctype, cconf, reasons = challenge_type, 1.0, ["type supplied by caller"]
        else:
            cls = classify_image(gray)
            ctype, cconf, reasons = cls.type, cls.confidence, cls.reasons

        result = PipelineResult(ctype, cconf, detail={"classifier_reasons": reasons})
        if ctype == "unsupported":
            result.error = "unsupported_challenge"
            result.total_ms = (time.perf_counter() - t0) * 1000
            return result

        quality = assess(gray)
        result.detail["quality"] = quality.metrics
        if not quality.ok:
            result.error = f"poor_quality: {quality.reason}"
            result.total_ms = round((time.perf_counter() - t0) * 1000, 2)
            return result
        gray = quality.gray

        best: dict | None = None
        solvers = [s for s in self._solvers(ctype) if allow_external or not getattr(s, "external", False)]
        result.detail["external_allowed"] = allow_external
        for solver in solvers:
            attempt = {"method": solver.name, "text": None, "confidence": 0.0,
                       "ms": 0.0, "cost_usd": 0.0, "error": None, "accepted": False}
            s0 = time.perf_counter()
            try:
                res = _POOL.submit(solver.solve, gray).result(timeout=self.timeout_s)
                attempt.update(text=res.text, confidence=round(res.confidence, 4),
                               cost_usd=res.cost_usd)
                attempt["detail"] = res.detail
                limit = self._threshold_for(solver)
                if limit is None:
                    attempt["error"] = "advisory_only"
                elif res.confidence < limit:
                    attempt["error"] = "low_confidence"
                elif (why := self._implausible(ctype, res.text)) is not None:
                    attempt["error"] = why
                else:
                    attempt["accepted"] = True
            except FutureTimeout:
                attempt["error"] = "timeout"
            except SolverUnavailable as exc:
                attempt["error"] = f"unavailable: {exc}"
            except Exception as exc:  # noqa: BLE001
                attempt["error"] = redact(f"{type(exc).__name__}: {exc}")
                # A call that was paid for but gave nothing usable still counts towards the cost.
                attempt["cost_usd"] = float(getattr(exc, "cost_usd", 0.0) or 0.0)
            attempt["ms"] = round((time.perf_counter() - s0) * 1000, 2)
            result.attempts.append(attempt)
            result.cost_usd += attempt["cost_usd"]
            logger.info("solver=%s accepted=%s error=%s conf=%.3f",
                        solver.name, attempt["accepted"], attempt["error"], attempt["confidence"])

            if attempt["accepted"]:
                best = attempt
                break
            if attempt["text"] is not None:
                # A plausible answer outranks a more confident one that already failed the shape check
                rank = (self._implausible(ctype, attempt["text"]) is None, attempt["confidence"])
                if best is None or rank > best_rank:
                    best, best_rank = attempt, rank

        if best is not None:
            result.text = best["text"]
            result.confidence = best["confidence"]
            result.method = best["method"]
            result.accepted = best["accepted"]
            result.detail.update(best.get("detail") or {})
            if not best["accepted"]:
                result.error = best["error"]
        else:
            result.error = "all_solvers_failed"
        result.total_ms = round((time.perf_counter() - t0) * 1000, 2)
        return result


def record_attempt(
    db: Session,
    res: PipelineResult,
    expected: str | None = None,
    filename: str | None = None,
) -> Attempt:
    """
    Persist one pipeline run.

    `success` means: the answer equals `expected` when a label is known,
    otherwise that a solver produced an answer above the confidence threshold.
    """
    success = (res.text == expected) if expected is not None else res.accepted
    row = Attempt(
        challenge_type=res.challenge_type,
        image_filename=filename,
        methods_tried=json.dumps([a["method"] for a in res.attempts]),
        method_used=res.method,
        predicted_text=res.text,
        expected_text=expected,
        success=bool(success),
        confidence=res.confidence,
        processing_ms=res.total_ms,
        cost_usd=res.cost_usd,
        error=redact(res.error) if res.error else None,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row
