"""Primary solver: the CRNN trained in ml/train_crnn.py, run with ONNX Runtime."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from app.config import settings
from app.solvers.base import SolveResult, SolverUnavailable
from ml.common import ctc_greedy_decode, preprocess


class CRNNSolver:
    name = "crnn_onnx"

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = Path(model_path or settings.crnn_model_path)
        self._sess = None
        self._lock = threading.Lock()

    def _session(self):
        with self._lock:     # create once; InferenceSession.run itself is thread-safe
            if self._sess is None:
                if not self.model_path.exists():
                    raise SolverUnavailable(
                        f"model not found at {self.model_path}; run ml/train_crnn.py"
                    )
                import onnxruntime as ort

                self._sess = ort.InferenceSession(
                    str(self.model_path), providers=["CPUExecutionProvider"]
                )
            return self._sess

    def solve(self, gray: np.ndarray) -> SolveResult:
        t0 = time.perf_counter()
        x = preprocess(gray)[None]
        logits = self._session().run(None, {"image": x})[0][0]
        text, conf = ctc_greedy_decode(logits)
        return SolveResult(
            text=text,
            confidence=conf,
            method=self.name,
            ms=(time.perf_counter() - t0) * 1000,
        )
