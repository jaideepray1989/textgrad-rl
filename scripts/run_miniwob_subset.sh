#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${MINIWOB_PYTHON:-.venv_miniwob/bin/python}"
OUT="${1:-runs/miniwob_subset}"

if [ -z "${MINIWOB_URL:-}" ]; then
  export MINIWOB_URL="file:///tmp/miniwob-plusplus/miniwob/html/miniwob/"
fi

"$PYTHON_BIN" -m textgrad_rl.benchmarks.miniwob_subset \
  --output-dir "$OUT" \
  --envs "${MINIWOB_ENVS:-default}" \
  --test-seeds "${MINIWOB_TEST_SEEDS:-3}" \
  --max-steps "${MINIWOB_MAX_STEPS:-5}" \
  --actor "${MINIWOB_ACTOR:-heuristic}" \
  --llm-base-url "${TEXTGRAD_RL_LLM_BASE_URL:-http://localhost:11434/v1}" \
  --model "${TEXTGRAD_RL_LLM_MODEL:-gpt-oss:20b}" \
  --temperature "${TEXTGRAD_RL_TEMPERATURE:-0.7}" \
  --llm-max-tokens "${MINIWOB_LLM_MAX_TOKENS:-256}" \
  --llm-timeout "${MINIWOB_LLM_TIMEOUT:-180}"
