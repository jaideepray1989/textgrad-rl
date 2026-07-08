#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python3}"
fi

"$PYTHON_BIN" -m textgrad_rl.benchmarks.textworld_express_suite \
  --train-seeds "${TEXTWORLD_TRAIN_SEEDS:-3}" \
  --val-seeds "${TEXTWORLD_VAL_SEEDS:-3}" \
  --test-seeds "${TEXTWORLD_TEST_SEEDS:-3}" \
  --max-steps "${TEXTWORLD_MAX_STEPS:-80}" \
  --output-dir "${TEXTWORLD_OUTPUT_DIR:-$ROOT_DIR/runs/textworld_express_suite}"
