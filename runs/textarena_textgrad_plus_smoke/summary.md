# TextArena TextGrad-RL+ Suite

- Environments: 2
- Methods: fixed_prompt, scalar_prompt_search, modular_textgrad, textgrad_rl_plus

## Implemented Improvements

- Multi-candidate textual optimization: targeted and generic candidates per causal assignment.
- Causal credit assignment: failed trajectories target environment-specific text variables.
- Replay validation: candidates are scored on train replay plus validation seeds.
- Uncertainty-aware gate: bootstrap delta CI with a minimum mean-gain threshold.
- Learned rule library: accepted rules are stored and retrieved into final test prompts.

## Test Means Across Repetitions

| method | reward | success | invalid | turns | accepted | candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt | 0.089 | 0.333 | 0.000 | 9.667 | 0.0 | 0.0 |
| scalar_prompt_search | 0.089 | 0.333 | 0.000 | 9.667 | 1.0 | 1.0 |
| modular_textgrad | 0.711 | 0.667 | 0.000 | 9.889 | 1.0 | 1.0 |
| textgrad_rl_plus | 0.756 | 0.667 | 0.000 | 9.333 | 1.0 | 6.0 |

## Bootstrap 95% CIs

| method | metric | mean | ci_low | ci_high | n |
| --- | --- | ---: | ---: | ---: | ---: |
| fixed_prompt | reward | 0.089 | -0.444 | 0.622 | 9 |
| fixed_prompt | success | 0.333 | 0.111 | 0.667 | 9 |
| fixed_prompt | invalid_move | 0.000 | 0.000 | 0.000 | 9 |
| modular_textgrad | reward | 0.711 | 0.444 | 0.911 | 9 |
| modular_textgrad | success | 0.667 | 0.333 | 1.000 | 9 |
| modular_textgrad | invalid_move | 0.000 | 0.000 | 0.000 | 9 |
| scalar_prompt_search | reward | 0.089 | -0.444 | 0.600 | 9 |
| scalar_prompt_search | success | 0.333 | 0.000 | 0.667 | 9 |
| scalar_prompt_search | invalid_move | 0.000 | 0.000 | 0.000 | 9 |
| textgrad_rl_plus | reward | 0.756 | 0.511 | 0.933 | 9 |
| textgrad_rl_plus | success | 0.667 | 0.333 | 1.000 | 9 |
| textgrad_rl_plus | invalid_move | 0.000 | 0.000 | 0.000 | 9 |
