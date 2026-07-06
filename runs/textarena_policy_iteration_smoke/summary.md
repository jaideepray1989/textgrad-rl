# TextGrad Policy Iteration Suite

- Environments: 2
- Methods: fixed_prompt, modular_textgrad, textgrad_rl_plus, textgrad_policy_iteration

## RL Strengthening

- Action-level credit assignment from trajectory actions, invalid moves, repeated actions, turn efficiency, and returns.
- Advantage-weighted textual gradients targeted at the worst negative-advantage actions.
- Candidate prompt policy search over a text-policy population.
- Replay buffer containing train, replay, validation, action-credit, and candidate-evaluation artifacts.
- Tabular value critic estimating environment baselines and candidate advantages.

## Test Means Across Repetitions

| method | reward | success | invalid | turns | accepted | candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt | 0.089 | 0.333 | 0.000 | 9.667 | 0.0 | 0.0 |
| modular_textgrad | 0.711 | 0.667 | 0.000 | 9.889 | 1.0 | 1.0 |
| textgrad_rl_plus | 0.756 | 0.667 | 0.000 | 9.333 | 1.0 | 6.0 |
| textgrad_policy_iteration | 0.756 | 0.667 | 0.000 | 9.333 | 1.0 | 8.0 |

## Bootstrap 95% CIs

| method | metric | mean | ci_low | ci_high | n |
| --- | --- | ---: | ---: | ---: | ---: |
| fixed_prompt | reward | 0.089 | -0.444 | 0.622 | 9 |
| fixed_prompt | success | 0.333 | 0.111 | 0.667 | 9 |
| fixed_prompt | invalid_move | 0.000 | 0.000 | 0.000 | 9 |
| modular_textgrad | reward | 0.711 | 0.444 | 0.911 | 9 |
| modular_textgrad | success | 0.667 | 0.333 | 1.000 | 9 |
| modular_textgrad | invalid_move | 0.000 | 0.000 | 0.000 | 9 |
| textgrad_policy_iteration | reward | 0.756 | 0.489 | 1.000 | 9 |
| textgrad_policy_iteration | success | 0.667 | 0.333 | 1.000 | 9 |
| textgrad_policy_iteration | invalid_move | 0.000 | 0.000 | 0.000 | 9 |
| textgrad_rl_plus | reward | 0.756 | 0.511 | 0.933 | 9 |
| textgrad_rl_plus | success | 0.667 | 0.333 | 1.000 | 9 |
| textgrad_rl_plus | invalid_move | 0.000 | 0.000 | 0.000 | 9 |
