#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python3}"
fi

MODE="${1:---preflight}"
OUTPUT_DIR="${OFFICIAL_PREFLIGHT_OUTPUT_DIR:-$ROOT_DIR/runs/official_leaderboard_preflight}"

case "$MODE" in
  --preflight)
    "$PYTHON_BIN" -m textgrad_rl.benchmarks.official_leaderboard_preflight \
      --suite tau2 \
      --output-dir "$OUTPUT_DIR"
    ;;
  --launch)
    "$PYTHON_BIN" -m textgrad_rl.benchmarks.official_leaderboard_preflight \
      --suite tau2 \
      --output-dir "$OUTPUT_DIR" \
      --strict

    IFS=',' read -ra DOMAINS <<< "${TAU2_DOMAINS:-retail,airline,telecom,banking_knowledge}"
    for raw_domain in "${DOMAINS[@]}"; do
      domain="$(echo "$raw_domain" | xargs)"
      [[ -z "$domain" ]] && continue
      save_to="${TAU2_OUTPUT_PREFIX:-textgrad_rl}_${domain}"
      if [[ "$domain" == "banking_knowledge" ]]; then
        tau2 run \
          --domain "$domain" \
          --retrieval-config "${TAU2_BANKING_RETRIEVAL_CONFIG:-alltools}" \
          --agent-llm "${TAU2_AGENT_LLM:-gpt-4.1}" \
          --user-llm "${TAU2_USER_LLM:-gpt-4.1}" \
          --num-trials "${TAU2_NUM_TRIALS:-4}" \
          --save-to "$save_to"
      else
        tau2 run \
          --domain "$domain" \
          --agent-llm "${TAU2_AGENT_LLM:-gpt-4.1}" \
          --user-llm "${TAU2_USER_LLM:-gpt-4.1}" \
          --num-trials "${TAU2_NUM_TRIALS:-4}" \
          --save-to "$save_to"
      fi
    done

    if [[ "${TAU2_PREPARE_SUBMISSION:-0}" == "1" ]]; then
      mapfile -t simulation_dirs < <(
        for raw_domain in "${DOMAINS[@]}"; do
          domain="$(echo "$raw_domain" | xargs)"
          [[ -z "$domain" ]] && continue
          echo "data/simulations/${TAU2_OUTPUT_PREFIX:-textgrad_rl}_${domain}"
        done
      )
      tau2 submit prepare "${simulation_dirs[@]}" --output "${TAU2_SUBMISSION_DIR:-./textgrad_rl_tau2_submission}"
    fi
    ;;
  *)
    echo "Usage: $0 [--preflight|--launch]" >&2
    exit 2
    ;;
esac
