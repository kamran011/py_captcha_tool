"""
GET /api/v1/solvers - which solvers are active, what each costs, and why one is off.

This is how a reviewer sees that the optional third-party solver is skipped cleanly
without a key. It never returns the key, only whether one is configured.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.limits import guard
from app.api.solve import get_pipeline
from app.solvers.pipeline import SolvePipeline
from app.solvers.vision_api_solver import describe

router = APIRouter(prefix="/api/v1", tags=["solvers"], dependencies=[Depends(guard)])


@router.get("/solvers")
def solvers(pipeline: SolvePipeline = Depends(get_pipeline)):
    items = {s["name"]: s for s in pipeline.solvers_status("text")}
    vision = describe()
    if vision["name"] in items:                                   # enabled: merge with its pipeline role
        items[vision["name"]].update({k: v for k, v in vision.items() if k not in ("name", "kind")})
    else:                                                          # disabled: still listed, with the reason
        items[vision["name"]] = {**vision, "challenge_types": ["text"], "threshold": None, "advisory_only": True}
    items["math_knn"] = pipeline.solvers_status("math")[0]
    for s in items.values():
        s.setdefault("enabled", True)
        s.setdefault("cost", "free" if s["kind"] == "local" else None)
    return {"solvers": list(items.values()),
            "note": "External solvers receive the image. They are last in the text pipeline and can be refused "
                    "per request with allow_external=false."}
