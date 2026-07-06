# TextGrad-RL PPO Ablation Results

This run compares five prompt-policy update rules across the deterministic TextArena policy suite, the difficulty-generalization suite, and the stochastic SLM suites.

Methods:

- `textgrad_rl_no_gate`: TextGrad-RL with no acceptance gate.
- `textgrad_rl_train_val`: TextGrad-RL with train/validation acceptance.
- `textgrad_rl_kl_gate`: TextGrad-RL with KL-style trust-region acceptance only.
- `textgrad_rl_clipped_surrogate`: TextGrad-RL with clipped-surrogate acceptance only.
- `textgrad_rl_ppo`: full TextGrad-RL-PPO gate: train/val improvement plus clipped surrogate, KL limit, and invalid/truncation safeguards.

SLM methods use the same gates with the `_slm` suffix.

## Runs

Deterministic policy suite:

```bash
.venv/bin/python -m textgrad_rl.benchmarks.textarena_policy_iteration \
  --methods textgrad_rl_no_gate,textgrad_rl_train_val,textgrad_rl_kl_gate,textgrad_rl_clipped_surrogate,textgrad_rl_ppo \
  --output-dir runs/textarena_policy_ablation_full
```

Expanded suite:

```bash
.venv/bin/python -m textgrad_rl.benchmarks.textarena_expanded_suites \
  --suites difficulty,puzzle,social,real_slm \
  --difficulty-methods textgrad_rl_no_gate,textgrad_rl_train_val,textgrad_rl_kl_gate,textgrad_rl_clipped_surrogate,textgrad_rl_ppo \
  --slm-methods textgrad_rl_no_gate_slm,textgrad_rl_train_val_slm,textgrad_rl_kl_gate_slm,textgrad_rl_clipped_surrogate_slm,textgrad_rl_ppo_slm \
  --slm-train-seeds 5 \
  --slm-val-seeds 5 \
  --slm-test-seeds 5 \
  --output-dir runs/textarena_ablation_full_suites_qwen25_3b_t07_5seed \
  --model qwen2.5:3b \
  --temperature 0.7 \
  --timeout 90
```

The SLM ablation uses `qwen2.5:3b` because this is the fast/free local model used for the five-seed stochastic run. `gpt-oss:20b` is installed, but the earlier one-seed gpt-oss SLM run took about 49 minutes and showed a high empty-output/action-format failure rate, so a five-seed full ablation would be a separate long experiment.

## Deterministic Policy Suite

10 environments, 3 repetitions, 390 test episodes per method.

| method | reward | success | invalid | turns | accepted | candidates |
|---|---:|---:|---:|---:|---:|---:|
| `textgrad_rl_no_gate` | 0.568 | 0.615 | 0.000 | 10.310 | 10.0 | 40.0 |
| `textgrad_rl_train_val` | 0.583 | 0.615 | 0.000 | 10.156 | 6.0 | 40.0 |
| `textgrad_rl_kl_gate` | 0.581 | 0.615 | 0.000 | 10.156 | 6.3 | 40.0 |
| `textgrad_rl_clipped_surrogate` | 0.581 | 0.615 | 0.000 | 10.156 | 6.3 | 40.0 |
| `textgrad_rl_ppo` | 0.583 | 0.615 | 0.000 | 10.156 | 6.0 | 40.0 |

Bootstrap reward CIs overlap strongly:

| method | reward mean | 95% CI |
|---|---:|---:|
| `textgrad_rl_no_gate` | 0.568 | [0.504, 0.629] |
| `textgrad_rl_train_val` | 0.583 | [0.521, 0.639] |
| `textgrad_rl_kl_gate` | 0.581 | [0.516, 0.641] |
| `textgrad_rl_clipped_surrogate` | 0.581 | [0.519, 0.640] |
| `textgrad_rl_ppo` | 0.583 | [0.521, 0.645] |

Takeaway: the train/val gate removes harmful prompt edits and gives the main deterministic lift. KL and clipped surrogate checks do not separate much because the deterministic candidates already remain close to the old policy. Full PPO ties train/val here.

## Difficulty Generalization

10 harder held-out environments, 2 repetitions, 36 test episodes per repetition.

| method | reward | success | invalid | turns |
|---|---:|---:|---:|---:|
| `textgrad_rl_no_gate` | 0.860 | 0.833 | 0.000 | 17.944 |
| `textgrad_rl_train_val` | 0.897 | 0.833 | 0.000 | 16.958 |
| `textgrad_rl_kl_gate` | 0.897 | 0.833 | 0.000 | 16.958 |
| `textgrad_rl_clipped_surrogate` | 0.897 | 0.833 | 0.000 | 16.958 |
| `textgrad_rl_ppo` | 0.897 | 0.833 | 0.000 | 16.958 |

Takeaway: all gated variants converge to the same learned rule library on this split. The no-gate variant is lower because it accepts every edit, including edits that do not improve validation behavior.

## SLM Suites

All SLM suites use `qwen2.5:3b`, temperature `0.7`, 5 train seeds, 5 validation seeds, and 5 test seeds per environment.

### Puzzle SLM

4 environments, 20 test episodes per method.

| method | reward | score | success | invalid | trunc | turns | update accepted |
|---|---:|---:|---:|---:|---:|---:|---:|
| `textgrad_rl_no_gate_slm` | 0.089 | -0.427 | 0.000 | 1.000 | 0.000 | 3.250 | true |
| `textgrad_rl_train_val_slm` | 0.109 | -0.406 | 0.000 | 1.000 | 0.000 | 3.100 | false |
| `textgrad_rl_kl_gate_slm` | 0.097 | -0.417 | 0.000 | 1.000 | 0.000 | 2.900 | true |
| `textgrad_rl_clipped_surrogate_slm` | 0.075 | -0.416 | 0.000 | 0.950 | 0.000 | 3.150 | false |
| `textgrad_rl_ppo_slm` | 0.129 | -0.387 | 0.000 | 1.000 | 0.000 | 3.300 | false |

### Social SLM

3 environments, 15 test episodes per method.

| method | reward | score | success | invalid | trunc | turns | update accepted |
|---|---:|---:|---:|---:|---:|---:|---:|
| `textgrad_rl_no_gate_slm` | -0.267 | -0.386 | 0.200 | 0.267 | 0.000 | 17.200 | true |
| `textgrad_rl_train_val_slm` | -0.400 | -0.583 | 0.133 | 0.333 | 0.000 | 16.667 | false |
| `textgrad_rl_kl_gate_slm` | -0.267 | -0.350 | 0.333 | 0.333 | 0.000 | 16.733 | false |
| `textgrad_rl_clipped_surrogate_slm` | -0.467 | -0.620 | 0.200 | 0.333 | 0.000 | 17.333 | false |
| `textgrad_rl_ppo_slm` | -0.333 | -0.419 | 0.267 | 0.267 | 0.000 | 17.133 | false |

### Real SLM

5 environments, 25 test episodes per method.

| method | reward | score | success | invalid | trunc | turns | update accepted |
|---|---:|---:|---:|---:|---:|---:|---:|
| `textgrad_rl_no_gate_slm` | 0.027 | -0.284 | 0.000 | 0.440 | 0.240 | 6.320 | true |
| `textgrad_rl_train_val_slm` | 0.413 | 0.412 | 0.320 | 0.160 | 0.200 | 6.360 | false |
| `textgrad_rl_kl_gate_slm` | 0.373 | 0.342 | 0.280 | 0.160 | 0.240 | 6.280 | false |
| `textgrad_rl_clipped_surrogate_slm` | 0.305 | 0.174 | 0.200 | 0.320 | 0.160 | 6.240 | false |
| `textgrad_rl_ppo_slm` | 0.194 | 0.061 | 0.160 | 0.240 | 0.240 | 6.720 | false |

## Gate Diagnostics

Each SLM method proposed one prompt update per suite. Deltas are candidate minus old policy on the train/validation gate samples.

| suite | method | accepted | train score delta | val score delta | surrogate delta | KL | clip frac | invalid delta | trunc delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| puzzle | `textgrad_rl_no_gate_slm` | true | -0.030 | -0.013 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| puzzle | `textgrad_rl_train_val_slm` | false | -0.011 | 0.000 | -0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| puzzle | `textgrad_rl_kl_gate_slm` | true | -0.006 | 0.030 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| puzzle | `textgrad_rl_clipped_surrogate_slm` | false | 0.019 | 0.002 | -0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| puzzle | `textgrad_rl_ppo_slm` | false | 0.012 | -0.013 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 |
| social | `textgrad_rl_no_gate_slm` | true | 0.100 | -0.295 | -0.088 | 0.059 | 0.300 | 0.100 | 0.000 |
| social | `textgrad_rl_train_val_slm` | false | -0.203 | -0.268 | -0.100 | 0.040 | 0.233 | 0.033 | 0.000 |
| social | `textgrad_rl_kl_gate_slm` | false | 0.335 | -0.168 | -0.138 | 0.065 | 0.367 | 0.033 | 0.000 |
| social | `textgrad_rl_clipped_surrogate_slm` | false | -0.406 | -0.402 | -0.129 | 0.077 | 0.467 | 0.033 | 0.000 |
| social | `textgrad_rl_ppo_slm` | false | 0.299 | -0.164 | -0.148 | 0.078 | 0.400 | 0.067 | 0.000 |
| real | `textgrad_rl_no_gate_slm` | true | 0.081 | -0.227 | -0.057 | 0.028 | 0.140 | 0.060 | -0.020 |
| real | `textgrad_rl_train_val_slm` | false | -0.221 | -0.200 | -0.033 | 0.040 | 0.220 | 0.020 | 0.060 |
| real | `textgrad_rl_kl_gate_slm` | false | 0.149 | -0.389 | -0.078 | 0.031 | 0.200 | 0.080 | -0.040 |
| real | `textgrad_rl_clipped_surrogate_slm` | false | 0.277 | -0.106 | -0.063 | 0.034 | 0.240 | 0.060 | -0.020 |
| real | `textgrad_rl_ppo_slm` | false | 0.173 | -0.090 | -0.026 | 0.043 | 0.220 | 0.080 | 0.000 |

## Interpretation

The ablation does not prove TextGrad-RL-PPO is better than TextGrad-RL on this full suite. It shows something narrower and useful:

1. The train/val gate is the strongest single stabilizer. It explains nearly all deterministic gains and prevents obviously harmful SLM updates.
2. KL-only is weak in these runs. The measured KL values are tiny for puzzle and below the target for social/real, so KL alone rarely rejects candidates that fail behaviorally.
3. Clipped surrogate is a meaningful stochastic diagnostic. In social and real SLM, rejected updates have negative surrogate deltas and large clip fractions, matching the intuition that sampled prompt edits are brittle.
4. Full PPO is conservative. It correctly rejects many SLM edits that hurt validation score or increase invalid moves, but it does not yet improve test reward reliably over the train/val gate.
5. To make a stronger paper claim, the next experiment should increase candidate population per update, train across multiple update rounds, and evaluate paired seeds with confidence intervals. The current full PPO gate is better framed as a safety/stability mechanism than as a demonstrated reward-improving algorithm.

## Artifacts

- `runs/textarena_policy_ablation_full/summary.md`
- `runs/textarena_policy_ablation_full/metrics_by_run.csv`
- `runs/textarena_policy_ablation_full/bootstrap_cis.csv`
- `runs/textarena_ablation_full_suites_qwen25_3b_t07_5seed/difficulty_generalization/summary.md`
- `runs/textarena_ablation_full_suites_qwen25_3b_t07_5seed/puzzle_slm/summary.md`
- `runs/textarena_ablation_full_suites_qwen25_3b_t07_5seed/social_slm/summary.md`
- `runs/textarena_ablation_full_suites_qwen25_3b_t07_5seed/real_slm/summary.md`
- `runs/textarena_ablation_full_suites_qwen25_3b_t07_5seed/*/update_decisions.jsonl`
