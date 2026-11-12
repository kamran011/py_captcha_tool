# Tasks ("The Executor")

| # | Milestone | Done when | Est. | Status |
|---|---|---|---|---|
| M0 | Scaffold: venv, folders, `requirements.txt`, config, redacting logger, SQLite models, FastAPI `/health`, Streamlit shell | API boots, `/health` returns 200 with DB ok, tests pass | 30 m | done |
| M1 | Synthetic dataset generator, CRNN training, ONNX export | `ml/evaluate.py` prints held-out exact-match and character accuracy | 90 m | done (97.4% held-out) |
| M2 | Solvers, fallback pipeline, image classifier, math type | Unit tests pass; a low-confidence case falls through to the next solver | 60 m | done |
| M3 | Provider detector | Fixture HTML for each provider and variant classified correctly | 30 m | done |
| M4 | Verification service, three demo pages, Playwright token script | Pass, invalid, reused, expired and missing each return the expected structure | 75 m | done |
| M5 | API endpoints and Streamlit pages (Solve, Verify, Overview) | Upload-solve, verify and metrics work in the browser | 60 m | done |
| M6 | Batch accuracy; CSV import with detection-only scan | Accuracy report and confusion matrix render in the dashboard | 45 m | done |
| M7 | Optional vision adapter with cost | Skipped cleanly with no key; cost shown when set | 30 m | done (offline-verified; live call needs a key) |
| M8 | Tests, README, architecture note, cost table, video script | A fresh clone runs from the README alone | 45 m | todo |

## M0 checklist

- [ ] `python -m venv .venv` and `pip install -r requirements.txt`
- [ ] Folder structure per `design.md`
- [ ] `app/config.py` — pydantic-settings, `SecretStr` for every secret
- [ ] `app/logging_config.py` — redacting filter and token fingerprint
- [ ] `app/db.py`, `app/models.py` — SQLite init on startup
- [ ] `app/main.py` — lifespan, `/health`, `/api/v1/stats`
- [ ] `dashboard/streamlit_app.py` — connects to the API, shows health and counters
- [ ] `tests/` — health, redaction, config
- [ ] `.env.example`, `.gitignore`, starter `README.md`
