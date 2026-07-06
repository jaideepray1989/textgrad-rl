#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python3}"
if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON" -m venv .venv
fi
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q
.venv/bin/python -m textgrad_rl.run_experiment --config configs/mac_demo.json
