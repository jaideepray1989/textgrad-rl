# Official Leaderboard Preflight Results

Status after adding and testing the official runner scaffolding:

| Suite | Can Launch Now | Main Blockers |
|---|---:|---|
| WebArena/WorkArena browser | no | official WebArena checkout/site URLs/API key, WorkArena package/ServiceNow credentials, Playwright runtime |
| tau2-bench | no | tau2 CLI/package and LLM runtime credentials |
| SWE-bench Lite | no | Docker, SWE-bench package, and predictions JSONL |

These are expected blockers on the current local machine. The scripts now make them explicit and will refuse `--launch` until the required harness dependencies and credentials are present.

Commands run:

```bash
bash scripts/run_official_browser_benchmarks.sh --preflight
bash scripts/run_official_taubench.sh --preflight
bash scripts/run_official_swebench_lite.sh --preflight
.venv/bin/python -m textgrad_rl.benchmarks.official_leaderboard_preflight --suite all --output-dir runs/official_leaderboard_preflight
.venv/bin/python -m pytest -q
```

The combined preflight artifact was written to `runs/official_leaderboard_preflight/official_leaderboard_preflight.md`.
