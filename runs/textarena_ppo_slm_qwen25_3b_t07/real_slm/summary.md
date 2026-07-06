# TextArena real_slm SLM Suite

- Model: qwen2.5:3b
- Temperature: 0.7

| method | reward | score | success | invalid | truncated | turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | 0.180 | 0.106 | 0.200 | 0.200 | 0.200 | 4.800 |
| textgrad_policy_iteration_slm | -0.020 | -0.255 | 0.000 | 0.200 | 0.400 | 7.000 |
| textgrad_ppo_slm | 0.180 | 0.104 | 0.200 | 0.200 | 0.200 | 5.200 |

## Per Environment

| method | env | reward | score | success | invalid | truncated | turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | GuessTheNumber-v0 | 1.000 | 1.490 | 1.000 | 0.000 | 0.000 | 2.000 |
| fixed_prompt_slm | FrozenLake-v0 | 0.500 | 0.485 | 0.000 | 0.000 | 0.000 | 3.000 |
| fixed_prompt_slm | Blackjack-v0 | 0.400 | 0.375 | 0.000 | 0.000 | 0.000 | 5.000 |
| fixed_prompt_slm | Nim-v0 | -1.000 | -1.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| fixed_prompt_slm | Mastermind-v0 | 0.000 | -0.310 | 0.000 | 0.000 | 1.000 | 12.000 |
| textgrad_policy_iteration_slm | GuessTheNumber-v0 | 0.000 | -0.310 | 0.000 | 0.000 | 1.000 | 12.000 |
| textgrad_policy_iteration_slm | FrozenLake-v0 | 0.500 | 0.480 | 0.000 | 0.000 | 0.000 | 4.000 |
| textgrad_policy_iteration_slm | Blackjack-v0 | 0.400 | 0.375 | 0.000 | 0.000 | 0.000 | 5.000 |
| textgrad_policy_iteration_slm | Nim-v0 | -1.000 | -1.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| textgrad_policy_iteration_slm | Mastermind-v0 | 0.000 | -0.310 | 0.000 | 0.000 | 1.000 | 12.000 |
| textgrad_ppo_slm | GuessTheNumber-v0 | 1.000 | 1.490 | 1.000 | 0.000 | 0.000 | 2.000 |
| textgrad_ppo_slm | FrozenLake-v0 | 0.500 | 0.485 | 0.000 | 0.000 | 0.000 | 3.000 |
| textgrad_ppo_slm | Blackjack-v0 | 0.400 | 0.375 | 0.000 | 0.000 | 0.000 | 5.000 |
| textgrad_ppo_slm | Nim-v0 | -1.000 | -1.520 | 0.000 | 1.000 | 0.000 | 4.000 |
| textgrad_ppo_slm | Mastermind-v0 | 0.000 | -0.310 | 0.000 | 0.000 | 1.000 | 12.000 |
