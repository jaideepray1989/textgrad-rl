# TextArena TextGrad-RL+ Suite

- Environments: 10
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
| fixed_prompt | 0.185 | 0.205 | 0.105 | 12.064 | 0.0 | 0.0 |
| scalar_prompt_search | 0.185 | 0.205 | 0.105 | 12.064 | 1.0 | 1.0 |
| modular_textgrad | 0.563 | 0.615 | 0.000 | 10.708 | 1.0 | 1.0 |
| textgrad_rl_plus | 0.583 | 0.615 | 0.000 | 10.156 | 6.0 | 30.0 |

## Bootstrap 95% CIs

| method | metric | mean | ci_low | ci_high | n |
| --- | --- | ---: | ---: | ---: | ---: |
| fixed_prompt | reward | 0.185 | 0.118 | 0.245 | 390 |
| fixed_prompt | success | 0.205 | 0.162 | 0.246 | 390 |
| fixed_prompt | invalid_move | 0.105 | 0.074 | 0.138 | 390 |
| modular_textgrad | reward | 0.563 | 0.503 | 0.623 | 390 |
| modular_textgrad | success | 0.615 | 0.567 | 0.662 | 390 |
| modular_textgrad | invalid_move | 0.000 | 0.000 | 0.000 | 390 |
| scalar_prompt_search | reward | 0.185 | 0.124 | 0.245 | 390 |
| scalar_prompt_search | success | 0.205 | 0.167 | 0.246 | 390 |
| scalar_prompt_search | invalid_move | 0.105 | 0.072 | 0.133 | 390 |
| textgrad_rl_plus | reward | 0.583 | 0.524 | 0.641 | 390 |
| textgrad_rl_plus | success | 0.615 | 0.567 | 0.662 | 390 |
| textgrad_rl_plus | invalid_move | 0.000 | 0.000 | 0.000 | 390 |
