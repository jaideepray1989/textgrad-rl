# TextGrad Policy Iteration Suite

- Environments: 10
- Methods: textgrad_rl_no_gate, textgrad_rl_train_val, textgrad_rl_kl_gate, textgrad_rl_clipped_surrogate, textgrad_rl_ppo

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
| textgrad_rl_no_gate | 0.568 | 0.615 | 0.000 | 10.310 | 10.0 | 40.0 |
| textgrad_rl_train_val | 0.583 | 0.615 | 0.000 | 10.156 | 6.0 | 40.0 |
| textgrad_rl_kl_gate | 0.581 | 0.615 | 0.000 | 10.156 | 6.3 | 40.0 |
| textgrad_rl_clipped_surrogate | 0.581 | 0.615 | 0.000 | 10.156 | 6.3 | 40.0 |
| textgrad_rl_ppo | 0.583 | 0.615 | 0.000 | 10.156 | 6.0 | 40.0 |

## Bootstrap 95% CIs

| method | metric | mean | ci_low | ci_high | n |
| --- | --- | ---: | ---: | ---: | ---: |
| textgrad_rl_clipped_surrogate | reward | 0.581 | 0.519 | 0.640 | 390 |
| textgrad_rl_clipped_surrogate | success | 0.615 | 0.567 | 0.662 | 390 |
| textgrad_rl_clipped_surrogate | invalid_move | 0.000 | 0.000 | 0.000 | 390 |
| textgrad_rl_kl_gate | reward | 0.581 | 0.516 | 0.641 | 390 |
| textgrad_rl_kl_gate | success | 0.615 | 0.567 | 0.659 | 390 |
| textgrad_rl_kl_gate | invalid_move | 0.000 | 0.000 | 0.000 | 390 |
| textgrad_rl_no_gate | reward | 0.568 | 0.504 | 0.629 | 390 |
| textgrad_rl_no_gate | success | 0.615 | 0.567 | 0.659 | 390 |
| textgrad_rl_no_gate | invalid_move | 0.000 | 0.000 | 0.000 | 390 |
| textgrad_rl_ppo | reward | 0.583 | 0.521 | 0.645 | 390 |
| textgrad_rl_ppo | success | 0.615 | 0.569 | 0.664 | 390 |
| textgrad_rl_ppo | invalid_move | 0.000 | 0.000 | 0.000 | 390 |
| textgrad_rl_train_val | reward | 0.583 | 0.521 | 0.639 | 390 |
| textgrad_rl_train_val | success | 0.615 | 0.567 | 0.664 | 390 |
| textgrad_rl_train_val | invalid_move | 0.000 | 0.000 | 0.000 | 390 |
