# TextArena social SLM Suite

- Model: qwen2.5:3b
- Temperature: 0.7

| method | reward | score | success | invalid | truncated | turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | -0.467 | -0.586 | 0.200 | 0.267 | 0.000 | 17.200 |
| textgrad_policy_iteration_slm | -0.733 | -0.985 | 0.067 | 0.400 | 0.000 | 17.000 |
| textgrad_ppo_slm | -0.600 | -0.785 | 0.067 | 0.267 | 0.000 | 17.067 |

## Per Environment

| method | env | reward | score | success | invalid | truncated | turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | KuhnPoker-v0-short | -1.000 | -1.422 | 0.000 | 0.800 | 0.000 | 4.400 |
| fixed_prompt_slm | IteratedPrisonersDilemma-v0 | -0.800 | -1.000 | 0.000 | 0.000 | 0.000 | 40.000 |
| fixed_prompt_slm | SimpleNegotiation-v0-short | 0.400 | 0.664 | 0.600 | 0.000 | 0.000 | 7.200 |
| textgrad_policy_iteration_slm | KuhnPoker-v0-short | -1.000 | -1.519 | 0.000 | 1.000 | 0.000 | 3.800 |
| textgrad_policy_iteration_slm | IteratedPrisonersDilemma-v0 | -0.800 | -1.000 | 0.000 | 0.000 | 0.000 | 40.000 |
| textgrad_policy_iteration_slm | SimpleNegotiation-v0-short | -0.400 | -0.436 | 0.200 | 0.200 | 0.000 | 7.200 |
| textgrad_ppo_slm | KuhnPoker-v0-short | -0.600 | -0.921 | 0.200 | 0.800 | 0.000 | 4.200 |
| textgrad_ppo_slm | IteratedPrisonersDilemma-v0 | -1.000 | -1.200 | 0.000 | 0.000 | 0.000 | 40.000 |
| textgrad_ppo_slm | SimpleNegotiation-v0-short | -0.200 | -0.235 | 0.000 | 0.000 | 0.000 | 7.000 |
