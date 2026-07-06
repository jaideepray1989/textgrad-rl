#!/usr/bin/env bash
set -euo pipefail
PYTHON="${TEXTGRAD_RL_PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON_FALLBACK:-python3}"
fi
"$PYTHON" -m textgrad_rl.run_experiment --config configs/mac_full.json
"$PYTHON" -m textgrad_rl.experiments.report --run-dir runs/mac_full
