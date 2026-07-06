# TextArena social SLM Suite

- Model: qwen2.5:3b
- Temperature: 0.7

| method | reward | score | success | invalid | truncated | turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| textgrad_rl_no_gate_slm | -0.267 | -0.386 | 0.200 | 0.267 | 0.000 | 17.200 |
| textgrad_rl_train_val_slm | -0.400 | -0.583 | 0.133 | 0.333 | 0.000 | 16.667 |
| textgrad_rl_kl_gate_slm | -0.267 | -0.350 | 0.333 | 0.333 | 0.000 | 16.733 |
| textgrad_rl_clipped_surrogate_slm | -0.467 | -0.620 | 0.200 | 0.333 | 0.000 | 17.333 |
| textgrad_rl_ppo_slm | -0.333 | -0.419 | 0.267 | 0.267 | 0.000 | 17.133 |

## Per Environment

| method | env | reward | score | success | invalid | truncated | turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| textgrad_rl_no_gate_slm | KuhnPoker-v0-short | -0.600 | -0.922 | 0.200 | 0.800 | 0.000 | 4.400 |
| textgrad_rl_no_gate_slm | IteratedPrisonersDilemma-v0 | -0.200 | -0.400 | 0.000 | 0.000 | 0.000 | 40.000 |
| textgrad_rl_no_gate_slm | SimpleNegotiation-v0-short | 0.000 | 0.164 | 0.400 | 0.000 | 0.000 | 7.200 |
| textgrad_rl_train_val_slm | KuhnPoker-v0-short | -1.000 | -1.515 | 0.000 | 1.000 | 0.000 | 3.000 |
| textgrad_rl_train_val_slm | IteratedPrisonersDilemma-v0 | -0.600 | -0.800 | 0.000 | 0.000 | 0.000 | 40.000 |
| textgrad_rl_train_val_slm | SimpleNegotiation-v0-short | 0.400 | 0.565 | 0.400 | 0.000 | 0.000 | 7.000 |
| textgrad_rl_kl_gate_slm | KuhnPoker-v0-short | -1.000 | -1.515 | 0.000 | 1.000 | 0.000 | 3.000 |
| textgrad_rl_kl_gate_slm | IteratedPrisonersDilemma-v0 | -0.600 | -0.700 | 0.200 | 0.000 | 0.000 | 40.000 |
| textgrad_rl_kl_gate_slm | SimpleNegotiation-v0-short | 0.800 | 1.164 | 0.800 | 0.000 | 0.000 | 7.200 |
| textgrad_rl_clipped_surrogate_slm | KuhnPoker-v0-short | -1.000 | -1.524 | 0.000 | 1.000 | 0.000 | 4.800 |
| textgrad_rl_clipped_surrogate_slm | IteratedPrisonersDilemma-v0 | -0.400 | -0.500 | 0.200 | 0.000 | 0.000 | 40.000 |
| textgrad_rl_clipped_surrogate_slm | SimpleNegotiation-v0-short | 0.000 | 0.164 | 0.400 | 0.000 | 0.000 | 7.200 |
| textgrad_rl_ppo_slm | KuhnPoker-v0-short | -1.000 | -1.422 | 0.000 | 0.800 | 0.000 | 4.400 |
| textgrad_rl_ppo_slm | IteratedPrisonersDilemma-v0 | -0.600 | -0.700 | 0.200 | 0.000 | 0.000 | 40.000 |
| textgrad_rl_ppo_slm | SimpleNegotiation-v0-short | 0.600 | 0.865 | 0.600 | 0.000 | 0.000 | 7.000 |
