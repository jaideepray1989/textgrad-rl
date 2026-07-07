#!/usr/bin/env bash
set -euo pipefail

ROOT="${WEBARENA_ROOT:-/tmp/webarena_inspect}"
OUT="${1:-runs/webarena_small_subset}"

.venv/bin/python -m textgrad_rl.benchmarks.webarena_subset \
  --webarena-root "$ROOT" \
  --output-dir "$OUT" \
  --task-count "${WEBARENA_TASK_COUNT:-20}" \
  --methods fixed_actor,textgrad_rl,textgrad_rl_ppo \
  --backend "${WEBARENA_BACKEND:-official}" \
  --model "${TEXTGRAD_RL_LLM_MODEL:-gpt-oss:20b}" \
  --temperature "${TEXTGRAD_RL_TEMPERATURE:-0.7}" \
  --max-steps "${WEBARENA_MAX_STEPS:-30}"
