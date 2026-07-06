# TextArena Paper Suite

- Environments: 10
- Methods: fixed_prompt, scalar_prompt_search, monolithic_textgrad, modular_textgrad, no_acceptance_gate

## Test Means Across Repetitions

| method | reward | success | invalid | turns |
| --- | ---: | ---: | ---: | ---: |
| fixed_prompt | 0.185 | 0.205 | 0.105 | 12.064 |
| scalar_prompt_search | 0.185 | 0.205 | 0.105 | 12.064 |
| monolithic_textgrad | 0.563 | 0.615 | 0.000 | 10.708 |
| modular_textgrad | 0.563 | 0.615 | 0.000 | 10.708 |
| no_acceptance_gate | 0.563 | 0.615 | 0.000 | 10.708 |

## Bootstrap 95% CIs

| method | metric | mean | ci_low | ci_high | n |
| --- | --- | ---: | ---: | ---: | ---: |
| fixed_prompt | reward | 0.185 | 0.118 | 0.245 | 390 |
| fixed_prompt | success | 0.205 | 0.162 | 0.246 | 390 |
| fixed_prompt | invalid_move | 0.105 | 0.074 | 0.138 | 390 |
| modular_textgrad | reward | 0.563 | 0.503 | 0.623 | 390 |
| modular_textgrad | success | 0.615 | 0.567 | 0.662 | 390 |
| modular_textgrad | invalid_move | 0.000 | 0.000 | 0.000 | 390 |
| monolithic_textgrad | reward | 0.563 | 0.500 | 0.623 | 390 |
| monolithic_textgrad | success | 0.615 | 0.567 | 0.659 | 390 |
| monolithic_textgrad | invalid_move | 0.000 | 0.000 | 0.000 | 390 |
| no_acceptance_gate | reward | 0.563 | 0.499 | 0.622 | 390 |
| no_acceptance_gate | success | 0.615 | 0.567 | 0.662 | 390 |
| no_acceptance_gate | invalid_move | 0.000 | 0.000 | 0.000 | 390 |
| scalar_prompt_search | reward | 0.185 | 0.124 | 0.245 | 390 |
| scalar_prompt_search | success | 0.205 | 0.167 | 0.246 | 390 |
| scalar_prompt_search | invalid_move | 0.105 | 0.072 | 0.133 | 390 |
