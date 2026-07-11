#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python3}"
fi

"$PYTHON_BIN" -m textgrad_rl.benchmarks.textworld_24_suite \
  --max-steps "${TEXTWORLD_24_MAX_STEPS:-80}" \
  --min-mean-delta "${TEXTWORLD_24_MIN_MEAN_DELTA:-0.001}" \
  --output-dir "${TEXTWORLD_24_OUTPUT_DIR:-$ROOT_DIR/runs/textworld_24_suite}" \
  "$@"
