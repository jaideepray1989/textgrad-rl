# TextArena Paper Suite

- Environments: 10
- Methods: fixed_prompt, scalar_prompt_search, monolithic_textgrad, modular_textgrad, no_acceptance_gate

## Test Means Across Repetitions

| method | reward | success | invalid | turns |
| --- | ---: | ---: | ---: | ---: |
| fixed_prompt | 0.184 | 0.231 | 0.115 | 12.231 |
| scalar_prompt_search | 0.184 | 0.231 | 0.115 | 12.231 |
| monolithic_textgrad | 0.563 | 0.615 | 0.000 | 11.154 |
| modular_textgrad | 0.563 | 0.615 | 0.000 | 11.154 |
| no_acceptance_gate | 0.563 | 0.615 | 0.000 | 11.154 |

## Bootstrap 95% CIs

| method | metric | mean | ci_low | ci_high | n |
| --- | --- | ---: | ---: | ---: | ---: |
| fixed_prompt | reward | 0.184 | -0.070 | 0.421 | 26 |
| fixed_prompt | success | 0.231 | 0.077 | 0.385 | 26 |
| fixed_prompt | invalid_move | 0.115 | 0.000 | 0.231 | 26 |
| modular_textgrad | reward | 0.563 | 0.319 | 0.778 | 26 |
| modular_textgrad | success | 0.615 | 0.423 | 0.808 | 26 |
| modular_textgrad | invalid_move | 0.000 | 0.000 | 0.000 | 26 |
| monolithic_textgrad | reward | 0.563 | 0.311 | 0.786 | 26 |
| monolithic_textgrad | success | 0.615 | 0.423 | 0.808 | 26 |
| monolithic_textgrad | invalid_move | 0.000 | 0.000 | 0.000 | 26 |
| no_acceptance_gate | reward | 0.563 | 0.302 | 0.786 | 26 |
| no_acceptance_gate | success | 0.615 | 0.423 | 0.808 | 26 |
| no_acceptance_gate | invalid_move | 0.000 | 0.000 | 0.000 | 26 |
| scalar_prompt_search | reward | 0.184 | -0.057 | 0.420 | 26 |
| scalar_prompt_search | success | 0.231 | 0.077 | 0.423 | 26 |
| scalar_prompt_search | invalid_move | 0.115 | 0.000 | 0.231 | 26 |
