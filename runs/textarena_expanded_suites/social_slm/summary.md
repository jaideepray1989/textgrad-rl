# TextArena social SLM Suite

- Model: qwen2.5:3b

| method | reward | score | success | invalid | truncated | turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | -0.333 | -0.417 | 0.333 | 0.333 | 0.000 | 16.667 |
| scalar_prompt_search_slm | -0.333 | -0.417 | 0.333 | 0.333 | 0.000 | 16.667 |
| modular_textgrad_slm | -0.333 | -0.417 | 0.333 | 0.333 | 0.000 | 16.667 |
| textgrad_rl_plus_slm | -0.333 | -0.585 | 0.000 | 0.333 | 0.000 | 17.000 |
| textgrad_policy_iteration_slm | -0.333 | -0.585 | 0.000 | 0.333 | 0.000 | 17.000 |

## Per Environment

| method | env | reward | score | success | invalid | truncated | turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | KuhnPoker-v0-short | -1.000 | -1.515 | 0.000 | 1.000 | 0.000 | 3.000 |
| fixed_prompt_slm | IteratedPrisonersDilemma-v0 | -1.000 | -1.200 | 0.000 | 0.000 | 0.000 | 40.000 |
| fixed_prompt_slm | SimpleNegotiation-v0-short | 1.000 | 1.465 | 1.000 | 0.000 | 0.000 | 7.000 |
| scalar_prompt_search_slm | KuhnPoker-v0-short | -1.000 | -1.515 | 0.000 | 1.000 | 0.000 | 3.000 |
| scalar_prompt_search_slm | IteratedPrisonersDilemma-v0 | -1.000 | -1.200 | 0.000 | 0.000 | 0.000 | 40.000 |
| scalar_prompt_search_slm | SimpleNegotiation-v0-short | 1.000 | 1.465 | 1.000 | 0.000 | 0.000 | 7.000 |
| modular_textgrad_slm | KuhnPoker-v0-short | -1.000 | -1.515 | 0.000 | 1.000 | 0.000 | 3.000 |
| modular_textgrad_slm | IteratedPrisonersDilemma-v0 | -1.000 | -1.200 | 0.000 | 0.000 | 0.000 | 40.000 |
| modular_textgrad_slm | SimpleNegotiation-v0-short | 1.000 | 1.465 | 1.000 | 0.000 | 0.000 | 7.000 |
| textgrad_rl_plus_slm | KuhnPoker-v0-short | -1.000 | -1.515 | 0.000 | 1.000 | 0.000 | 3.000 |
| textgrad_rl_plus_slm | IteratedPrisonersDilemma-v0 | 0.000 | -0.200 | 0.000 | 0.000 | 0.000 | 40.000 |
| textgrad_rl_plus_slm | SimpleNegotiation-v0-short | 0.000 | -0.040 | 0.000 | 0.000 | 0.000 | 8.000 |
| textgrad_policy_iteration_slm | KuhnPoker-v0-short | -1.000 | -1.515 | 0.000 | 1.000 | 0.000 | 3.000 |
| textgrad_policy_iteration_slm | IteratedPrisonersDilemma-v0 | 0.000 | -0.200 | 0.000 | 0.000 | 0.000 | 40.000 |
| textgrad_policy_iteration_slm | SimpleNegotiation-v0-short | 0.000 | -0.040 | 0.000 | 0.000 | 0.000 | 8.000 |
