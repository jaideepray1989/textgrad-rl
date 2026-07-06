# TextGrad-PPO Benchmark Comparison

This comparison runs every existing deterministic TextArena suite where `textgrad_ppo` is implemented:

1. Base TextArena policy-iteration benchmark over the 10 default environments.
2. Difficulty Generalization over harder TextArena variants.

The SLM suites are not included here because they use separate `*_slm` methods and do not yet have a `textgrad_ppo_slm` implementation.

## Commands

Base policy-iteration suite:

```bash
.venv/bin/python -m textgrad_rl.benchmarks.textarena_policy_iteration \
  --methods fixed_prompt,textgrad_policy_iteration,textgrad_ppo \
  --repetitions 3 \
  --train-seeds 5 \
  --val-seeds 5 \
  --test-seeds 10 \
  --output-dir runs/textarena_ppo_comparison_policy_iteration_corrected
```

Difficulty Generalization:

```bash
.venv/bin/python -m textgrad_rl.benchmarks.textarena_expanded_suites \
  --suites difficulty \
  --difficulty-methods fixed_prompt,textgrad_policy_iteration,textgrad_ppo \
  --repetitions 2 \
  --train-seeds 3 \
  --val-seeds 3 \
  --test-seeds 3 \
  --output-dir runs/textarena_ppo_comparison_expanded_difficulty_corrected
```

## Results

### Base TextArena Policy-Iteration Suite

| method | reward | success | invalid | turns | accepted | candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt | 0.185 | 0.205 | 0.105 | 12.064 | 0.0 | 0.0 |
| textgrad_policy_iteration | 0.583 | 0.615 | 0.000 | 10.156 | 6.0 | 40.0 |
| textgrad_ppo | 0.583 | 0.615 | 0.000 | 10.156 | 6.0 | 40.0 |

Bootstrap 95% CIs:

| method | metric | mean | ci_low | ci_high | n |
| --- | --- | ---: | ---: | ---: | ---: |
| fixed_prompt | reward | 0.185 | 0.118 | 0.245 | 390 |
| fixed_prompt | success | 0.205 | 0.162 | 0.246 | 390 |
| fixed_prompt | invalid_move | 0.105 | 0.074 | 0.138 | 390 |
| textgrad_policy_iteration | reward | 0.583 | 0.524 | 0.638 | 390 |
| textgrad_policy_iteration | success | 0.615 | 0.569 | 0.662 | 390 |
| textgrad_policy_iteration | invalid_move | 0.000 | 0.000 | 0.000 | 390 |
| textgrad_ppo | reward | 0.583 | 0.520 | 0.639 | 390 |
| textgrad_ppo | success | 0.615 | 0.569 | 0.664 | 390 |
| textgrad_ppo | invalid_move | 0.000 | 0.000 | 0.000 | 390 |

### Difficulty Generalization

| method | reward | success | invalid | turns |
| --- | ---: | ---: | ---: | ---: |
| fixed_prompt | 0.190 | 0.167 | 0.139 | 22.597 |
| textgrad_policy_iteration | 0.897 | 0.833 | 0.000 | 16.958 |
| textgrad_ppo | 0.897 | 0.833 | 0.000 | 16.958 |

## Interpretation

`textgrad_ppo` matches `textgrad_policy_iteration` on these deterministic benchmarks while strongly improving over `fixed_prompt`. It accepts the same six learned rules as policy iteration on the base suite:

- `Nim-v0`
- `LightsOut-v0`
- `FrozenLake-v0`
- `TowerOfHanoi-v0`
- `GuessTheNumber-v0`
- `Mastermind-v0`

The important implementation fix was to let the PPO surrogate use action-credit fallback advantages. Without that, uniformly bad training trajectories can have zero episode-level advantage even when action-level credit clearly identifies a bad behavior. With the fallback, PPO accepts the Tower of Hanoi recursive-rule update and generalizes to the harder Tower of Hanoi variants.

## Artifacts

- `runs/textarena_ppo_comparison_policy_iteration_corrected/summary.md`
- `runs/textarena_ppo_comparison_policy_iteration_corrected/metrics_by_run.csv`
- `runs/textarena_ppo_comparison_policy_iteration_corrected/per_env_metrics.csv`
- `runs/textarena_ppo_comparison_expanded_difficulty_corrected/difficulty_generalization/summary.md`
- `runs/textarena_ppo_comparison_expanded_difficulty_corrected/difficulty_generalization/metrics.csv`
- `runs/textarena_ppo_comparison_expanded_difficulty_corrected/difficulty_generalization/per_env_metrics.csv`
