# TextGrad Policy Iteration Suite

- Environments: 2
- Methods: fixed_prompt, modular_textgrad, textgrad_rl_plus, textgrad_policy_iteration, textgrad_ppo

## RL Strengthening

- Action-level credit assignment from trajectory actions, invalid moves, repeated actions, turn efficiency, and returns.
- Advantage-weighted textual gradients targeted at the worst negative-advantage actions.
- Candidate prompt policy search over a text-policy population.
- Replay buffer containing train, replay, validation, action-credit, and candidate-evaluation artifacts.
- Tabular value critic estimating environment baselines and candidate advantages.
- Text-PPO variant with clipped behavioral ratios, surrogate-delta acceptance, and KL-style trust-region limits.

## Test Means Across Repetitions

| method | reward | success | invalid | turns | accepted | candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt | 0.333 | 0.667 | 0.000 | 10.667 | 0.0 | 0.0 |
| modular_textgrad | 1.000 | 1.000 | 0.000 | 10.333 | 1.0 | 1.0 |
| textgrad_rl_plus | 1.000 | 1.000 | 0.000 | 10.333 | 1.0 | 3.0 |
| textgrad_policy_iteration | 1.000 | 1.000 | 0.000 | 10.333 | 1.0 | 4.0 |
| textgrad_ppo | 1.000 | 1.000 | 0.000 | 10.333 | 1.0 | 4.0 |

## Bootstrap 95% CIs

| method | metric | mean | ci_low | ci_high | n |
| --- | --- | ---: | ---: | ---: | ---: |
| fixed_prompt | reward | 0.333 | -1.000 | 1.000 | 3 |
| fixed_prompt | success | 0.667 | 0.000 | 1.000 | 3 |
| fixed_prompt | invalid_move | 0.000 | 0.000 | 0.000 | 3 |
| modular_textgrad | reward | 1.000 | 1.000 | 1.000 | 3 |
| modular_textgrad | success | 1.000 | 1.000 | 1.000 | 3 |
| modular_textgrad | invalid_move | 0.000 | 0.000 | 0.000 | 3 |
| textgrad_policy_iteration | reward | 1.000 | 1.000 | 1.000 | 3 |
| textgrad_policy_iteration | success | 1.000 | 1.000 | 1.000 | 3 |
| textgrad_policy_iteration | invalid_move | 0.000 | 0.000 | 0.000 | 3 |
| textgrad_ppo | reward | 1.000 | 1.000 | 1.000 | 3 |
| textgrad_ppo | success | 1.000 | 1.000 | 1.000 | 3 |
| textgrad_ppo | invalid_move | 0.000 | 0.000 | 0.000 | 3 |
| textgrad_rl_plus | reward | 1.000 | 1.000 | 1.000 | 3 |
| textgrad_rl_plus | success | 1.000 | 1.000 | 1.000 | 3 |
| textgrad_rl_plus | invalid_move | 0.000 | 0.000 | 0.000 | 3 |
