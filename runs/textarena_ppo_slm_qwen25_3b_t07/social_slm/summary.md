# TextArena social SLM Suite

- Model: qwen2.5:3b
- Temperature: 0.7

| method | reward | score | success | invalid | truncated | turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | -0.333 | -0.417 | 0.333 | 0.333 | 0.000 | 16.667 |
| textgrad_policy_iteration_slm | 0.000 | -0.083 | 0.333 | 0.333 | 0.000 | 16.667 |
| textgrad_ppo_slm | -0.333 | -0.417 | 0.333 | 0.333 | 0.000 | 16.667 |

## Per Environment

| method | env | reward | score | success | invalid | truncated | turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | KuhnPoker-v0-short | -1.000 | -1.515 | 0.000 | 1.000 | 0.000 | 3.000 |
| fixed_prompt_slm | IteratedPrisonersDilemma-v0 | -1.000 | -1.200 | 0.000 | 0.000 | 0.000 | 40.000 |
| fixed_prompt_slm | SimpleNegotiation-v0-short | 1.000 | 1.465 | 1.000 | 0.000 | 0.000 | 7.000 |
| textgrad_policy_iteration_slm | KuhnPoker-v0-short | -1.000 | -1.515 | 0.000 | 1.000 | 0.000 | 3.000 |
| textgrad_policy_iteration_slm | IteratedPrisonersDilemma-v0 | 0.000 | -0.200 | 0.000 | 0.000 | 0.000 | 40.000 |
| textgrad_policy_iteration_slm | SimpleNegotiation-v0-short | 1.000 | 1.465 | 1.000 | 0.000 | 0.000 | 7.000 |
| textgrad_ppo_slm | KuhnPoker-v0-short | -1.000 | -1.515 | 0.000 | 1.000 | 0.000 | 3.000 |
| textgrad_ppo_slm | IteratedPrisonersDilemma-v0 | -1.000 | -1.200 | 0.000 | 0.000 | 0.000 | 40.000 |
| textgrad_ppo_slm | SimpleNegotiation-v0-short | 1.000 | 1.465 | 1.000 | 0.000 | 0.000 | 7.000 |
