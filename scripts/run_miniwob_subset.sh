#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${MINIWOB_PYTHON:-.venv_miniwob/bin/python}"
OUT="${1:-runs/miniwob_subset}"

if [ -z "${MINIWOB_URL:-}" ]; then
  export MINIWOB_URL="file:///tmp/miniwob-plusplus/miniwob/html/miniwob/"
fi

"$PYTHON_BIN" -m textgrad_rl.benchmarks.miniwob_subset \
  --output-dir "$OUT" \
  --test-seeds "${MINIWOB_TEST_SEEDS:-3}" \
  --max-steps "${MINIWOB_MAX_STEPS:-5}"
