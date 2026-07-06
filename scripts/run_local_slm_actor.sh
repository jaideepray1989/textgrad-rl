#!/usr/bin/env bash
set -euo pipefail
PYTHON="${TEXTGRAD_RL_PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON_FALLBACK:-python3}"
fi
export TEXTGRAD_RL_LLM_BASE_URL="${TEXTGRAD_RL_LLM_BASE_URL:-http://localhost:11434/v1}"
export TEXTGRAD_RL_LLM_MODEL="${TEXTGRAD_RL_LLM_MODEL:-qwen2.5-coder:3b}"
"$PYTHON" -m textgrad_rl.run_experiment \
  --agent local_llm \
  --critic heuristic \
  --method modular_textgrad \
  --iterations 3 \
  --train-tasks 10 \
  --val-tasks 5 \
  --test-tasks 5 \
  --output-dir runs/local_slm_actor
