# WebArena Second Benchmark Status

Goal: run a small WebArena subset with `fixed_actor`, `textgrad_rl`, and `textgrad_rl_ppo`, logging task success, invalid browser actions, repeated actions, turns, and validation-gated prompt updates.

## What is implemented

- Added `textgrad_rl.benchmarks.webarena_subset`.
- Selects a deterministic, site-balanced 20-task subset from the official WebArena `config_files/test.raw.json`.
- Writes the method matrix for:
  - `fixed_actor`
  - `textgrad_rl`
  - `textgrad_rl_ppo`
- Runs the official WebArena `ScriptBrowserEnv` loop when preflight passes.
- For `textgrad_rl`, trains on the first task slice, proposes a textual-gradient prompt edit, and accepts it only if validation score does not regress.
- For `textgrad_rl_ppo`, applies the same train/validation gate plus a clipped-surrogate and KL-proxy trust-region check before accepting the prompt edit.
- Emits canonical artifacts:
  - `config.json`
  - `task_subset.json`
  - `task_subset_summary.json`
  - `method_configs.json`
  - `preflight.json`
  - `episodes.jsonl`
  - `prompt_updates.jsonl`
  - `summary.csv`
  - `summary.md`

## Command run

```bash
scripts/run_webarena_small_subset.sh runs/webarena_small_subset_20
```

The command returned `2`, which is the harness code for `not_run_preflight_failed`.

## Current result

No WebArena task success scores were produced on this machine. The harness blocked before browser execution because the official WebArena infrastructure is not available locally.

Planned task mix:

| Site group | Tasks |
|---|---:|
| gitlab | 2 |
| gitlab+reddit | 2 |
| gitlab+wikipedia | 2 |
| map | 2 |
| map+shopping_admin | 2 |
| map+wikipedia | 1 |
| reddit | 2 |
| reddit+gitlab | 2 |
| shopping | 2 |
| shopping+reddit | 1 |
| shopping_admin | 1 |
| wikipedia+map | 1 |

Evaluation mix:

| Eval type | Count |
|---|---:|
| program_html | 9 |
| string_match | 9 |
| url_match | 5 |

Blocked summary:

| Method | Status | Tasks planned | Success rate | Invalid browser action rate | Repeated action rate | Avg turns | Validation-gated prompt updates |
|---|---|---:|---:|---:|---:|---:|---:|
| fixed_actor | blocked | 20 |  |  |  |  | 0 |
| textgrad_rl | blocked | 20 |  |  |  |  | 0 |
| textgrad_rl_ppo | blocked | 20 |  |  |  |  | 0 |

## Missing pieces

- `docker` is not installed or not on `PATH`.
- Official WebArena Python runtime dependencies are missing in the project venv: `beartype`, `gymnasium`, `playwright`, `tiktoken`.
- Official WebArena site URLs are not configured: `SHOPPING`, `SHOPPING_ADMIN`, `REDDIT`, `GITLAB`, `MAP`, `WIKIPEDIA`, `HOMEPAGE`.
- Generated per-task config files are missing for the selected task IDs.
- Login state files are missing under `.auth/`.
- Because those are missing, `episodes.jsonl` contains 60 `not_run_preflight_failed` rows rather than completed WebArena trajectories.

## To produce real paper results

1. Set up the official WebArena stack or the WebArena AMI.
2. Export the official WebArena URL variables.
3. Install WebArena requirements in a Python 3.10/3.11 environment and run `playwright install`.
4. Run `python scripts/generate_test_data.py` in the WebArena repo.
5. Run `python browser_env/auto_login.py` or `bash prepare.sh` to create `.auth/*_state.json`.
6. Re-run:

```bash
WEBARENA_ROOT=/path/to/webarena scripts/run_webarena_small_subset.sh runs/webarena_small_subset_20
```

The current harness will then execute the official WebArena episodes for the three methods and replace the blocked rows with completed task-level metrics.
