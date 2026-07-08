#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${TEXTGRAD_RL_ACTOR_MODEL:=qwen2.5:3b}"
: "${TEXTGRAD_RL_CANDIDATE_MODEL:=gpt-oss:20b}"
: "${TEXTGRAD_RL_LLM_BASE_URL:=http://localhost:11434/v1}"

# Temperature sensitivity: fixed actor only, same 30 test seeds as the t=0.7 run.
.venv/bin/python -m textgrad_rl.benchmarks.textarena_expanded_suites \
  --suites puzzle,social,real_slm \
  --slm-methods fixed_prompt_slm \
  --slm-train-seeds 30 \
  --slm-val-seeds 30 \
  --slm-test-seeds 30 \
  --slm-candidate-count 1 \
  --output-dir runs/textarena_slm_qwen25_3b_t00_30seed_fixed \
  --base-url "$TEXTGRAD_RL_LLM_BASE_URL" \
  --model "$TEXTGRAD_RL_ACTOR_MODEL" \
  --temperature 0.0 \
  --timeout 120

# Candidate-count ablation. The candidate_count=8 leg is already
# runs/textarena_slm_candidate_pool_30seed; this script fills the missing 1 and 4.
for candidate_count in 1 4; do
  .venv/bin/python -m textgrad_rl.benchmarks.textarena_expanded_suites \
    --suites puzzle,social,real_slm \
    --slm-methods fixed_prompt_slm,textgrad_rl_no_gate_slm,textgrad_rl_train_val_slm \
    --slm-train-seeds 30 \
    --slm-val-seeds 30 \
    --slm-test-seeds 30 \
    --slm-candidate-count "$candidate_count" \
    --slm-candidate-model "$TEXTGRAD_RL_CANDIDATE_MODEL" \
    --slm-candidate-temperature 0.2 \
    --output-dir "runs/textarena_slm_candidate_count_${candidate_count}_30seed" \
    --base-url "$TEXTGRAD_RL_LLM_BASE_URL" \
    --model "$TEXTGRAD_RL_ACTOR_MODEL" \
    --temperature 0.7 \
    --timeout 120
done

.venv/bin/python scripts/build_workshop_experimental_package.py
