# Official Leaderboard Execution

This repo separates local protocol probes from official leaderboard executions. Use these scripts to test readiness before launching heavyweight benchmark harnesses:

```bash
bash scripts/run_official_browser_benchmarks.sh --preflight
bash scripts/run_official_taubench.sh --preflight
bash scripts/run_official_swebench_lite.sh --preflight
```

Pass `--launch` only after preflight is clean.

## Browser: WebArena/WorkArena

```bash
export BROWSER_OFFICIAL_TARGETS=webarena,workarena
export WEBARENA_REPO=/path/to/webarena
export SHOPPING=...
export SHOPPING_ADMIN=...
export REDDIT=...
export GITLAB=...
export MAP=...
export WIKIPEDIA=...
export HOMEPAGE=...
export OPENAI_API_KEY=...
export SNOW_INSTANCE_URL=...
export SNOW_INSTANCE_UNAME=admin
export SNOW_INSTANCE_PWD='...'

bash scripts/run_official_browser_benchmarks.sh --launch
```

WebArena follows the official repo flow: generate test configs, create auto-login cookies, then run `run.py`. WorkArena follows the BrowserGym package flow and verifies that a ServiceNow task can reset. Set `WORKARENA_INITIALIZE=1` if the ServiceNow instance still needs `workarena-install`.

## tau2-bench

```bash
export OPENAI_API_KEY=...
export TAU2_AGENT_LLM=gpt-4.1
export TAU2_USER_LLM=gpt-4.1
export TAU2_NUM_TRIALS=4
export TAU2_DOMAINS=retail,airline,telecom,banking_knowledge

bash scripts/run_official_taubench.sh --launch
```

Set `TAU2_PREPARE_SUBMISSION=1` to run `tau2 submit prepare` after all selected domains finish.

## SWE-bench Lite

```bash
export SWE_BENCH_PREDICTIONS_PATH=/path/to/textgrad_rl_predictions.jsonl
export SWE_BENCH_MAX_WORKERS=4
export SWE_BENCH_RUN_ID=textgrad_rl_swebench_lite

bash scripts/run_official_swebench_lite.sh --launch
```

Use `SWE_BENCH_PREDICTIONS_PATH=gold` only to validate that the local harness works; it is not a model result.

## Current Local Status

The latest local readiness test is recorded in `OFFICIAL_LEADERBOARD_PREFLIGHT_RESULTS.md`. Full machine-readable details are written under `runs/official_leaderboard_preflight/`, which is intentionally ignored by git.
