# TextGrad Policy Iteration Suite

- Environments: 10
- Methods: fixed_prompt, textgrad_policy_iteration, textgrad_ppo

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
| fixed_prompt | 0.185 | 0.205 | 0.105 | 12.064 | 0.0 | 0.0 |
| textgrad_policy_iteration | 0.583 | 0.615 | 0.000 | 10.156 | 6.0 | 40.0 |
| textgrad_ppo | 0.583 | 0.615 | 0.000 | 10.156 | 6.0 | 40.0 |

## Bootstrap 95% CIs

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
