# TextGrad-PPO Prototype

This adds a PPO-style trust-region layer over TextGrad prompt updates.

TextGrad-PPO is not token-level PPO and does not fine-tune model weights. It treats text variables as the policy, uses the previous text policy as the behavior policy, evaluates candidate TextGrad edits on paired old/new rollouts with identical seeds, and accepts only candidates that pass:

- replay/bootstrap performance gating,
- clipped surrogate improvement,
- KL-style trust-region limit over a behavioral ratio proxy,
- no invalid-move-rate regression.

## Text-PPO Surrogate

Because prompt policies do not expose action log-probabilities, the implementation estimates a behavioral ratio from paired score deltas:

- positive-advantage old trajectory improved by the candidate: ratio increases;
- negative-advantage old trajectory improved by the candidate: bad-behavior ratio decreases;
- large moves are clipped with PPO epsilon before scoring the surrogate.

Default PPO settings:

| setting | value |
| --- | ---: |
| `clip_epsilon` | 0.2 |
| `target_kl` | 0.2 |
| `score_scale` | 5.0 |
| `min_surrogate_delta` | 0.001 |

## Smoke Run

Command:

```bash
.venv/bin/python -m textgrad_rl.benchmarks.textarena_policy_iteration \
  --envs GuessTheNumber-v0,Nim-v0 \
  --repetitions 1 \
  --train-seeds 1 \
  --val-seeds 1 \
  --test-seeds 1 \
  --output-dir runs/textarena_textgrad_ppo_smoke
```

Results:

| method | reward | success | invalid | turns | accepted | candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt | 0.333 | 0.667 | 0.000 | 10.667 | 0.0 | 0.0 |
| modular_textgrad | 1.000 | 1.000 | 0.000 | 10.333 | 1.0 | 1.0 |
| textgrad_rl_plus | 1.000 | 1.000 | 0.000 | 10.333 | 1.0 | 3.0 |
| textgrad_policy_iteration | 1.000 | 1.000 | 0.000 | 10.333 | 1.0 | 4.0 |
| textgrad_ppo | 1.000 | 1.000 | 0.000 | 10.333 | 1.0 | 4.0 |

Accepted PPO update:

| env | rule | mean_delta | surrogate_delta | approx_kl | clip_fraction | objective |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Nim-v0 | xor strategy | 1.2525 | 0.125 | 0.0966 | 0.500 | 0.3898 |

Artifacts:

- `runs/textarena_textgrad_ppo_smoke/summary.md`
- `runs/textarena_textgrad_ppo_smoke/methods/textgrad_ppo/rep_000/ppo_candidate_evaluations.jsonl`
- `runs/textarena_textgrad_ppo_smoke/methods/textgrad_ppo/rep_000/learned_rule_library.json`

## Next Full Run

```bash
.venv/bin/python -m textgrad_rl.benchmarks.textarena_policy_iteration \
  --repetitions 3 \
  --train-seeds 5 \
  --val-seeds 5 \
  --test-seeds 10 \
  --output-dir runs/textarena_policy_iteration_with_ppo
```

For difficulty generalization, `textgrad_ppo` is also available through `textarena_expanded_suites`.
