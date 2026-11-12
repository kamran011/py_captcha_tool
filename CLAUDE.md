@AGENTS.md

## Claude Code specifics

- Before editing UI code, run the dashboard and check the page in the browser pane; type checks and tests do not cover Streamlit layout.
- Terminal tabs the user opened (API, Streamlit) cannot be stopped from here. Ask the user to Ctrl+C them, then start a fresh process from a new terminal tab.
- Anything that spends money (`ml/calibrate.py --vision`, live vision tests) needs the user's explicit go-ahead and a stated worst-case cost first.
- Never read `.env` into a reply or a commit message; when a key is involved, report only whether it is configured.
