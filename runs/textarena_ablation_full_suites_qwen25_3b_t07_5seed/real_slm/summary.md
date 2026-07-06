# TextArena real_slm SLM Suite

- Model: qwen2.5:3b
- Temperature: 0.7

| method | reward | score | success | invalid | truncated | turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| textgrad_rl_no_gate_slm | 0.027 | -0.284 | 0.000 | 0.440 | 0.240 | 6.320 |
| textgrad_rl_train_val_slm | 0.413 | 0.412 | 0.320 | 0.160 | 0.200 | 6.360 |
| textgrad_rl_kl_gate_slm | 0.373 | 0.342 | 0.280 | 0.160 | 0.240 | 6.280 |
| textgrad_rl_clipped_surrogate_slm | 0.305 | 0.174 | 0.200 | 0.320 | 0.160 | 6.240 |
| textgrad_rl_ppo_slm | 0.194 | 0.061 | 0.160 | 0.240 | 0.240 | 6.720 |

## Per Environment

| method | env | reward | score | success | invalid | truncated | turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| textgrad_rl_no_gate_slm | GuessTheNumber-v0 | 0.389 | -0.005 | 0.000 | 0.600 | 0.200 | 8.800 |
| textgrad_rl_no_gate_slm | FrozenLake-v0 | 0.467 | 0.249 | 0.000 | 0.400 | 0.000 | 3.600 |
| textgrad_rl_no_gate_slm | Blackjack-v0 | 0.280 | 0.158 | 0.000 | 0.200 | 0.000 | 4.400 |
| textgrad_rl_no_gate_slm | Nim-v0 | -1.000 | -1.514 | 0.000 | 1.000 | 0.000 | 2.800 |
| textgrad_rl_no_gate_slm | Mastermind-v0 | 0.000 | -0.310 | 0.000 | 0.000 | 1.000 | 12.000 |
| textgrad_rl_train_val_slm | GuessTheNumber-v0 | 1.000 | 1.473 | 1.000 | 0.000 | 0.000 | 5.400 |
| textgrad_rl_train_val_slm | FrozenLake-v0 | 0.467 | 0.249 | 0.000 | 0.400 | 0.000 | 3.600 |
| textgrad_rl_train_val_slm | Blackjack-v0 | 0.400 | 0.375 | 0.000 | 0.000 | 0.000 | 5.000 |
| textgrad_rl_train_val_slm | Nim-v0 | 0.200 | 0.271 | 0.600 | 0.400 | 0.000 | 5.800 |
| textgrad_rl_train_val_slm | Mastermind-v0 | 0.000 | -0.310 | 0.000 | 0.000 | 1.000 | 12.000 |
| textgrad_rl_kl_gate_slm | GuessTheNumber-v0 | 0.800 | 1.118 | 0.800 | 0.000 | 0.200 | 6.400 |
| textgrad_rl_kl_gate_slm | FrozenLake-v0 | 0.467 | 0.249 | 0.000 | 0.400 | 0.000 | 3.600 |
| textgrad_rl_kl_gate_slm | Blackjack-v0 | 0.400 | 0.375 | 0.000 | 0.000 | 0.000 | 5.000 |
| textgrad_rl_kl_gate_slm | Nim-v0 | 0.200 | 0.278 | 0.600 | 0.400 | 0.000 | 4.400 |
| textgrad_rl_kl_gate_slm | Mastermind-v0 | 0.000 | -0.310 | 0.000 | 0.000 | 1.000 | 12.000 |
| textgrad_rl_clipped_surrogate_slm | GuessTheNumber-v0 | 0.684 | 0.798 | 0.600 | 0.200 | 0.200 | 7.200 |
| textgrad_rl_clipped_surrogate_slm | FrozenLake-v0 | 0.467 | 0.249 | 0.000 | 0.400 | 0.000 | 3.600 |
| textgrad_rl_clipped_surrogate_slm | Blackjack-v0 | 0.400 | 0.375 | 0.000 | 0.000 | 0.000 | 5.000 |
| textgrad_rl_clipped_surrogate_slm | Nim-v0 | -0.200 | -0.318 | 0.400 | 0.600 | 0.000 | 3.600 |
| textgrad_rl_clipped_surrogate_slm | Mastermind-v0 | 0.175 | -0.234 | 0.000 | 0.400 | 0.600 | 11.800 |
| textgrad_rl_ppo_slm | GuessTheNumber-v0 | 0.705 | 0.915 | 0.600 | 0.000 | 0.200 | 8.000 |
| textgrad_rl_ppo_slm | FrozenLake-v0 | 0.467 | 0.249 | 0.000 | 0.400 | 0.000 | 3.600 |
| textgrad_rl_ppo_slm | Blackjack-v0 | 0.400 | 0.375 | 0.000 | 0.000 | 0.000 | 5.000 |
| textgrad_rl_ppo_slm | Nim-v0 | -0.600 | -0.925 | 0.200 | 0.800 | 0.000 | 5.000 |
| textgrad_rl_ppo_slm | Mastermind-v0 | 0.000 | -0.310 | 0.000 | 0.000 | 1.000 | 12.000 |
