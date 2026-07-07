# Broad 50-Benchmark Completion Results

This report completes the two requested broad suites:

1. TextArena broad 50-environment offline suite.
2. BrowserGym MiniWoB++ 50-task suite.

## Commands

TextArena broad 50-environment suite:

```bash
.venv/bin/python -m textgrad_rl.benchmarks.textarena_broad_suite \
  --output-dir runs/textarena_broad_50 \
  --test-seeds 3 \
  --train-seeds 3 \
  --val-seeds 3 \
  --policy-training-test-seeds 1 \
  --turn-budget 80
```

BrowserGym MiniWoB++ 50-task suite:

```bash
MINIWOB_ENVS=50 scripts/run_miniwob_subset.sh runs/miniwob_subset_50x3
```

## TextArena 50-Environment Suite

The TextArena suite covers 50 offline registered TextArena environments: 35 single-player tasks and 15 two-player games. The run uses paired evaluation seeds across methods. Two-player games are evaluated from both target sides, producing 195 held-out test episodes per method.

The current TextGrad-RL policy actor has environment-specific rule logic for 19 environments or difficulty variants derived from the original 10 supported policy families. The other 31 environments use a generic legal-action fallback, and are reported separately.

| Method | Envs | Episodes | Reward | Success | Invalid | Repeated | Avg Turns | Updates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_prompt | 50 | 195 | 0.094 | 0.185 | 0.518 | 0.749 | 9.32 | 0 |
| textgrad_policy_iteration | 50 | 195 | 0.261 | 0.369 | 0.451 | 0.718 | 8.50 | 6 |

Supported-policy-family slice:

| Method | Envs | Episodes | Reward | Success | Invalid | Repeated | Avg Turns |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_prompt | 19 | 72 | 0.173 | 0.167 | 0.222 | 0.833 | 13.29 |
| textgrad_policy_iteration | 19 | 72 | 0.626 | 0.667 | 0.042 | 0.750 | 11.07 |

Generic-fallback slice:

| Method | Envs | Episodes | Reward | Success | Invalid | Repeated | Avg Turns |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_prompt | 31 | 123 | 0.048 | 0.195 | 0.691 | 0.699 | 7.00 |
| textgrad_policy_iteration | 31 | 123 | 0.048 | 0.195 | 0.691 | 0.699 | 7.00 |

The overall TextArena lift is driven by the supported policy-family slice. TextGrad-RL improves success from `0.167` to `0.667` on that slice, reduces invalid moves from `0.222` to `0.042`, and preserves the expected no-change result on generic fallback tasks where no environment-specific textual policy is available.

Per-environment success improvements under paired seeds:

| Env | fixed_prompt | textgrad_policy_iteration |
|---|---:|---:|
| FrozenLake-v0 | 0.000 | 1.000 |
| FrozenLake-v0-hardcore | 0.000 | 1.000 |
| FrozenLake-v0-random | 0.000 | 1.000 |
| GuessTheNumber-v0 | 0.000 | 1.000 |
| GuessTheNumber-v0-hardcore | 0.000 | 1.000 |
| LightsOut-v0 | 0.000 | 1.000 |
| Mastermind-v0 | 0.000 | 1.000 |
| Mastermind-v0-hard | 0.000 | 1.000 |
| TowerOfHanoi-v0 | 0.000 | 1.000 |
| TowerOfHanoi-v0-medium | 0.000 | 1.000 |
| Nim-v0 | 0.500 | 1.000 |
| Nim-v0-medium | 0.500 | 1.000 |

## BrowserGym MiniWoB++ 50-Task Suite

The MiniWoB++ suite covers 50 browser tasks with three test seeds per task, producing 150 held-out test episodes per method. Coordinate-heavy drag/draw tasks are intentionally excluded because this actor uses accessibility-tree actions.

| Method | Episodes | Success | Invalid Action | Repeated Action | Avg Turns | Accepted Updates |
|---|---:|---:|---:|---:|---:|---:|
| fixed_actor | 150 | 0.200 | 0.020 | 0.713 | 4.05 | 0 |
| textgrad_rl | 150 | 0.273 | 0.020 | 0.640 | 3.86 | 1 |
| textgrad_rl_ppo | 150 | 0.200 | 0.020 | 0.713 | 4.05 | 0 |

The MiniWoB++ gain is concentrated in selection tasks. TextGrad-RL learns the rule to click Submit after selecting checkbox/radio/list options, increasing selection success from `0.000` to `0.407` in the 50-task suite.

## Interpretation

These two suites establish a stronger broad-benchmark story than the original 10-game TextArena run:

- TextArena broad 50 shows that learned textual policy rules transfer across difficulty variants of supported policy families.
- BrowserGym MiniWoB++ 50 shows the same failure-to-text-rule loop can improve browser-control behavior outside TextArena.
- The main remaining limitation is clear: TextGrad-RL does not automatically solve arbitrary TextArena games unless the actor can execute the learned text policy in that action space. The generic-fallback slice is intentionally flat and should be reported as a boundary of the current prototype, not hidden in the aggregate.

## Artifacts

- `runs/textarena_broad_50/summary.md`
- `runs/textarena_broad_50/summary.csv`
- `runs/textarena_broad_50/slice_summary.csv`
- `runs/textarena_broad_50/per_env_metrics.csv`
- `runs/textarena_broad_50/fixed_prompt_episodes.jsonl`
- `runs/textarena_broad_50/textgrad_policy_iteration_episodes.jsonl`
- `runs/miniwob_subset_50x3/summary.md`
- `runs/miniwob_subset_50x3/summary.csv`
- `runs/miniwob_subset_50x3/category_summary.csv`
- `runs/miniwob_subset_50x3/episodes.jsonl`
