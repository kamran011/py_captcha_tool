# Demo video script (target 4:30, limit 5:00)

A reviewer sees the real tool, so every number below is one the tool itself shows. Say the caveats out loud; they are part of the answer, and the assignment weighs reliability and honest evaluation.

## Before recording (2 minutes)

```bash
# terminal 1
uvicorn app.main:app --port 8000
# terminal 2
streamlit run dashboard/streamlit_app.py
# terminal 3: confirms everything works; keep it open for the Verify segment
python scripts/smoke_test.py --online
```

- Open http://127.0.0.1:8501 and http://127.0.0.1:8000/docs in two tabs. Browser window about 1280 px wide.
- Needs internet only for the Verify segment (the providers' test widgets). If the network is down, skip to the mocked tests: `pytest tests/test_verification.py -q`.
- Do not show a token: the tool never prints one (fingerprint only), and neither should you.
- First solve after start is slower (it loads the model); do one before recording.
- Optional: clear old rows by deleting `data/captcha_tool.db` and restarting the API, so Overview starts at zero and fills as you go.

## Beats

| Time | Show | Say |
|---|---|---|
| **0:00 - 0:25** | README top, then `docs/architecture.md` diagram | "Assignment 01: recognise and solve CAPTCHAs in a test environment, and verify provider tokens server side. Per sections 1 and 7 I only use synthetic CAPTCHAs, local demo pages and the providers' official test keys. The 200-site CSV is a labelled benchmark that is scanned read-only; I did not build form submission or solving on third-party sites." |
| **0:25 - 1:25** | Dashboard **Solve** | Held-out sample: "A CRNN I trained on generated CAPTCHAs, served with ONNX Runtime. It reads this in a few milliseconds, at zero marginal cost." Math sample: "Arithmetic is read and evaluated with an AST whitelist, never `eval`." Grid sample: "Unsupported types are classified and refused, never sent to a solver." Held-out sample + Degrade **blur**: "A quality gate rejects unreadable input, because the model would otherwise answer confidently on garbage." |
| **1:25 - 2:20** | Terminal: `python scripts/acquire_token.py --provider recaptcha --replay`, then `--provider turnstile --scenario fail`; then dashboard **Verify** page history | "This drives a real test widget in a browser on my local demo page, gets a token, and verifies it server side with the provider. The second submission of the same token is rejected locally with 409 and never reaches Google; the ledger is written before the provider call, so concurrent replays cannot both pass. Turnstile's published always-fail secret gives a structured invalid-token result. Logs and the database hold only a fingerprint." Mention: missing, malformed, expired, reused and provider-down each return a distinct status; two outcomes (invalid reCAPTCHA token, expiry by age) are demonstrated by mocked tests only, because Google's test secret accepts anything. |
| **2:20 - 3:05** | **Detect** page (fixture), then **Sites** report | "Provider detection reads the HTML with a real parser and strict host matching. On the 200 supplied sites: one read-only GET per page, robots.txt obeyed. 87 of 200 pages could be analysed; the rest answered 403, even for robots.txt, and I did not work around that. On those 87: provider-level accuracy 77%, precision 100%, no false alarms. The misses are widgets injected by JavaScript, which a static, read-only scan cannot see, and I report that instead of hiding it." |
| **3:05 - 3:40** | **Batch** page: dataset `test`, 1000 images | "1,000 held-out images: 98.1% exact match, 99.6% character accuracy, p95 under 5 ms. 18 answers were wrong but confident: confidence does not flag the model's mistakes, which is why thresholds come from measured precision and why the fallback solver is advisory." Mention the caveat: accuracy is on my own generator. |
| **3:40 - 4:10** | **Overview** "Solvers and cost" table, then `docs/cost.md` | "Default cost is zero per image. An optional vision-API solver exists for a last resort: it is off without a key, shows why, sends the image to the provider only when allowed, prices each call from reported token usage, and has a daily budget and rate cap. I could not test the real provider path without a key, so that part is verified against a mock only, and there is an opt-in live test ready." |
| **4:10 - 4:35** | `pytest -q` output or `docs/architecture.md` limitations | "517 tests pass offline. Safeguards were mutation-checked: disabling replay protection, the scanner's private-address block, or the budget guard makes tests fail. Known limits: accuracy on other CAPTCHA styles is untested, rotated or mirrored images can be accepted wrongly, and the guards are in-process, single-process by design." |
| **4:35 - 4:45** | README quick start | "Everything runs from the README: venv, install, copy the env file, start two commands." |

## If something goes wrong on camera

| Problem | Do |
|---|---|
| Provider widget does not load / no internet | skip the browser flow; run `pytest tests/test_verification.py -q` and show the replay and failure matrix |
| Playwright cannot find Chromium | `python -m playwright install chromium` (once) |
| Port busy | `uvicorn app.main:app --port 8001` and set `API_BASE=http://127.0.0.1:8001` for the dashboard |
| Overview shows zeros | it fills as you solve and verify; run the smoke test first to seed it |
