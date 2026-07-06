#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${TEXTGRAD_RL_ACTOR_MODEL:=qwen2.5:3b}"
: "${TEXTGRAD_RL_CANDIDATE_MODEL:=gpt-oss:20b}"
: "${TEXTGRAD_RL_LLM_BASE_URL:=http://localhost:11434/v1}"

.venv/bin/python -m textgrad_rl.benchmarks.textarena_expanded_suites \
  --suites puzzle,social,real_slm \
  --slm-methods fixed_prompt_slm,textgrad_rl_no_gate_slm,textgrad_rl_train_val_slm,textgrad_rl_ppo_slm \
  --slm-train-seeds 30 \
  --slm-val-seeds 30 \
  --slm-test-seeds 30 \
  --slm-candidate-count 8 \
  --slm-candidate-model "$TEXTGRAD_RL_CANDIDATE_MODEL" \
  --slm-candidate-temperature 0.2 \
  --output-dir runs/textarena_slm_candidate_pool_30seed \
  --base-url "$TEXTGRAD_RL_LLM_BASE_URL" \
  --model "$TEXTGRAD_RL_ACTOR_MODEL" \
  --temperature 0.7 \
  --timeout 120
