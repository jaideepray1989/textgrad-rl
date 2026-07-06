# TextGrad Policy Iteration Suite

- Environments: 10
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
| textgrad_policy_iteration | 0.583 | 0.615 | 0.000 | 10.156 | 6.0 | 40.0 |

## Bootstrap 95% CIs

| method | metric | mean | ci_low | ci_high | n |
| --- | --- | ---: | ---: | ---: | ---: |
| textgrad_policy_iteration | reward | 0.583 | 0.524 | 0.638 | 390 |
| textgrad_policy_iteration | success | 0.615 | 0.569 | 0.662 | 390 |
| textgrad_policy_iteration | invalid_move | 0.000 | 0.000 | 0.000 | 390 |
