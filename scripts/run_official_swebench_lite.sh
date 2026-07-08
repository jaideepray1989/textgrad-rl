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
      --suite swebench \
      --output-dir "$OUTPUT_DIR"
    ;;
  --launch)
    "$PYTHON_BIN" -m textgrad_rl.benchmarks.official_leaderboard_preflight \
      --suite swebench \
      --output-dir "$OUTPUT_DIR" \
      --strict

    namespace_args=()
    if [[ -n "${SWE_BENCH_NAMESPACE+x}" ]]; then
      namespace_args=(--namespace "$SWE_BENCH_NAMESPACE")
    elif [[ "$(uname -m)" == "arm64" || "$(uname -m)" == "aarch64" ]]; then
      namespace_args=(--namespace "")
    fi

    "$PYTHON_BIN" -m swebench.harness.run_evaluation \
      --dataset_name "${SWE_BENCH_DATASET_NAME:-princeton-nlp/SWE-bench_Lite}" \
      --predictions_path "${SWE_BENCH_PREDICTIONS_PATH:?Set SWE_BENCH_PREDICTIONS_PATH to a predictions JSONL, or gold for harness validation only.}" \
      --max_workers "${SWE_BENCH_MAX_WORKERS:-4}" \
      --run_id "${SWE_BENCH_RUN_ID:-textgrad_rl_swebench_lite}" \
      "${namespace_args[@]}"
    ;;
  *)
    echo "Usage: $0 [--preflight|--launch]" >&2
    exit 2
    ;;
esac
