# CAPTCHA Solving & Verification Tool

Technology Brainz, Assignment 01. FastAPI backend (`app/`), Streamlit dashboard (`dashboard/`), custom CRNN model (`ml/`), SQLite database. Python 3.12 on Windows; the venv is `.venv`.

## Hard limits (assignment sections 1 and 7: never relax these)

- Work only on synthetic CAPTCHAs, local demo pages (`web/`) and the providers' published test keys.
- Never solve or bypass anti-bot protection on a third-party site. Never fill or submit a contact form, and never automate unsolicited submissions.
- The site scan is read-only: one plain GET per page plus `robots.txt` (obeyed). Nothing is filled, submitted or solved.
- Tokens and secrets are never printed or logged; at most a short fingerprint. `.env` is git-ignored: never commit it, never echo its contents.

## Commands

Use `.venv\Scripts\python.exe` (or activate the venv); the system Python lacks the dependencies.

```bash
.venv/Scripts/python.exe -m pytest -q                 # about 45 s, fully offline; live tests are opt-in (RUN_LIVE=1)
.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
.venv/Scripts/python.exe -m streamlit run dashboard/streamlit_app.py   # http://127.0.0.1:8501
.venv/Scripts/python.exe scripts/smoke_test.py        # --online also checks provider test endpoints
.venv/Scripts/python.exe ml/calibrate.py              # per-solver acceptance thresholds
.venv/Scripts/python.exe ml/calibrate.py --vision N   # SPENDS REAL MONEY (see below)
```

API calls need the header `X-API-Key` (default `changeme-local`, from `.env`). In PowerShell use `Invoke-RestMethod`; `curl` there is an alias for `Invoke-WebRequest` and rejects Unix flags.

## Layout

- `app/solvers/pipeline.py`: classify, quality gate, then solvers in order: `crnn_onnx`, `segment_knn` (advisory only), optional `vision_api`. Every attempt is recorded.
- `app/solvers/vision_api_solver.py`: optional third-party LLM reader, last tier. Off unless `VISION_API_KEY` is set.
- `app/verification/`, `app/sites/`: token verification and the read-only provider detector.
- `ml/`: dataset generation, training, evaluation, calibration. `data/models/calibration.json` is read by the pipeline.
- `dashboard/samples.py`: demo inputs, including the hard-sample tiers. Sample indices 0-19 are shown in the UI; calibration uses indices from 100 up.
- `docs/`: `architecture.md`, `design.md`, `cost.md`, `requirements.md`, `demo-script.md`. Numbers in the docs are measured; keep each one tagged measured / derived / estimate / quoted.

## Gotchas

- **Settings come from `.env`.** Variable names: `VISION_API_KEY`, `VISION_PROVIDER` (`anthropic` | `openai`), `VISION_MODEL`, `VISION_PRICE_IN_PER_MTOK`, `VISION_PRICE_OUT_PER_MTOK`, `VISION_BUDGET_USD_PER_DAY`. A model missing from the built-in price table needs both price variables, otherwise the solver refuses to run.
- **Tests never read `.env`.** `tests/conftest.py` points `CAPTCHA_ENV_FILE` at the null device and strips `VISION_*`, so a real key can never reach a paid API from the suite. Keep it that way.
- **Restart after changing `.env` or the key.** Settings, the vision circuit breaker (opens on HTTP 401/403) and the budget guard are in-process state.
- **Vision calibration costs money** and sends images to the provider. Run it only with an explicit N and the user's go-ahead; the budget guard (`VISION_BUDGET_USD_PER_DAY`) is the ceiling. A threshold is applied only when the calibration entry's recorded `model` matches `VISION_MODEL`.
- **A null threshold means advisory:** the solver runs and is shown but its answer is never accepted alone. `segment_knn` and, measured so far, `vision_api` are advisory.
- **The database persists.** `data/captcha_tool.db` accumulates every test and manual run (the Overview page reads it). Clear the runtime tables (`attempts`, `verifications`, `token_ledger`, `batch_runs`) for a clean demo, but keep `sites`.
- **Do not show wrong answers as right.** A confident wrong answer must stay visible as wrong in the UI; do not tune the display or the data to flatter results.

## Conventions

- Default to no comments; add one only for a non-obvious why. Prefer editing existing files.
- Add or update a test for every behaviour change; the suite must stay green before any commit.
- Update the docs in the same change when a measured number changes (`README.md` results table, `docs/cost.md`).
- Never commit unless asked. Never use `--no-verify`.
