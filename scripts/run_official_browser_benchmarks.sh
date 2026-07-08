#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python3}"
fi

MODE="${1:---preflight}"
OUTPUT_DIR="${OFFICIAL_PREFLIGHT_OUTPUT_DIR:-$ROOT_DIR/runs/official_leaderboard_preflight}"
TARGETS="${BROWSER_OFFICIAL_TARGETS:-webarena,workarena}"

case "$MODE" in
  --preflight)
    "$PYTHON_BIN" -m textgrad_rl.benchmarks.official_leaderboard_preflight \
      --suite browser \
      --browser-targets "$TARGETS" \
      --output-dir "$OUTPUT_DIR"
    ;;
  --launch)
    "$PYTHON_BIN" -m textgrad_rl.benchmarks.official_leaderboard_preflight \
      --suite browser \
      --browser-targets "$TARGETS" \
      --output-dir "$OUTPUT_DIR" \
      --strict

    IFS=',' read -ra TARGET_ARRAY <<< "$TARGETS"
    for raw_target in "${TARGET_ARRAY[@]}"; do
      target="$(echo "$raw_target" | tr '[:upper:]' '[:lower:]' | xargs)"
      case "$target" in
        webarena)
          : "${WEBARENA_REPO:?Set WEBARENA_REPO to the official WebArena checkout.}"
          (
            cd "$WEBARENA_REPO"
            python scripts/generate_test_data.py
            mkdir -p ./.auth
            python browser_env/auto_login.py
            python run.py \
              --instruction_path "${WEBARENA_INSTRUCTION_PATH:-agent/prompts/jsons/p_cot_id_actree_2s.json}" \
              --test_start_idx "${WEBARENA_TEST_START_IDX:-0}" \
              --test_end_idx "${WEBARENA_TEST_END_IDX:-1}" \
              --model "${WEBARENA_MODEL:-gpt-4.1}" \
              --result_dir "${WEBARENA_RESULT_DIR:-$ROOT_DIR/runs/official_webarena}"
          )
          ;;
        workarena)
          if [[ "${WORKARENA_INITIALIZE:-0}" == "1" ]]; then
            workarena-install
          fi
          "$PYTHON_BIN" - <<'PY'
import gymnasium as gym
import browsergym.workarena  # noqa: F401

task_id = __import__("os").environ.get("WORKARENA_TASK_ID", "browsergym/workarena.servicenow.filter-asset-list")
env = gym.make(task_id)
obs, info = env.reset()
env.close()
print(f"WorkArena smoke reset succeeded for {task_id}")
PY
          ;;
        "")
          ;;
        *)
          echo "Unknown BROWSER_OFFICIAL_TARGETS entry: $target" >&2
          exit 2
          ;;
      esac
    done
    ;;
  *)
    echo "Usage: $0 [--preflight|--launch]" >&2
    exit 2
    ;;
esac
