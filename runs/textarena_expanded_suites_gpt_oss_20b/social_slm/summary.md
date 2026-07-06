# TextArena social SLM Suite

- Model: gpt-oss:20b

| method | reward | score | success | invalid | truncated | turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | 0.333 | 0.413 | 0.333 | 0.000 | 0.000 | 17.333 |
| scalar_prompt_search_slm | 0.333 | 0.413 | 0.333 | 0.000 | 0.000 | 17.333 |
| modular_textgrad_slm | 0.333 | 0.418 | 0.333 | 0.000 | 0.000 | 16.333 |
| textgrad_rl_plus_slm | 0.333 | 0.413 | 0.333 | 0.000 | 0.000 | 17.333 |
| textgrad_policy_iteration_slm | 0.333 | 0.413 | 0.333 | 0.000 | 0.000 | 17.333 |

## Per Environment

| method | env | reward | score | success | invalid | truncated | turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | KuhnPoker-v0-short | 1.000 | 1.475 | 1.000 | 0.000 | 0.000 | 5.000 |
| fixed_prompt_slm | IteratedPrisonersDilemma-v0 | 0.000 | -0.200 | 0.000 | 0.000 | 0.000 | 40.000 |
| fixed_prompt_slm | SimpleNegotiation-v0-short | 0.000 | -0.035 | 0.000 | 0.000 | 0.000 | 7.000 |
| scalar_prompt_search_slm | KuhnPoker-v0-short | 1.000 | 1.475 | 1.000 | 0.000 | 0.000 | 5.000 |
| scalar_prompt_search_slm | IteratedPrisonersDilemma-v0 | 0.000 | -0.200 | 0.000 | 0.000 | 0.000 | 40.000 |
| scalar_prompt_search_slm | SimpleNegotiation-v0-short | 0.000 | -0.035 | 0.000 | 0.000 | 0.000 | 7.000 |
| modular_textgrad_slm | KuhnPoker-v0-short | 1.000 | 1.490 | 1.000 | 0.000 | 0.000 | 2.000 |
| modular_textgrad_slm | IteratedPrisonersDilemma-v0 | 0.000 | -0.200 | 0.000 | 0.000 | 0.000 | 40.000 |
| modular_textgrad_slm | SimpleNegotiation-v0-short | 0.000 | -0.035 | 0.000 | 0.000 | 0.000 | 7.000 |
| textgrad_rl_plus_slm | KuhnPoker-v0-short | 1.000 | 1.475 | 1.000 | 0.000 | 0.000 | 5.000 |
| textgrad_rl_plus_slm | IteratedPrisonersDilemma-v0 | 0.000 | -0.200 | 0.000 | 0.000 | 0.000 | 40.000 |
| textgrad_rl_plus_slm | SimpleNegotiation-v0-short | 0.000 | -0.035 | 0.000 | 0.000 | 0.000 | 7.000 |
| textgrad_policy_iteration_slm | KuhnPoker-v0-short | 1.000 | 1.475 | 1.000 | 0.000 | 0.000 | 5.000 |
| textgrad_policy_iteration_slm | IteratedPrisonersDilemma-v0 | 0.000 | -0.200 | 0.000 | 0.000 | 0.000 | 40.000 |
| textgrad_policy_iteration_slm | SimpleNegotiation-v0-short | 0.000 | -0.035 | 0.000 | 0.000 | 0.000 | 7.000 |
