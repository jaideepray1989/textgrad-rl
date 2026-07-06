# TextArena real_slm SLM Suite

- Model: qwen2.5:3b
- Temperature: 0.7

| method | reward | score | success | invalid | truncated | turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | 0.380 | 0.245 | 0.200 | 0.320 | 0.160 | 7.120 |
| textgrad_policy_iteration_slm | 0.377 | 0.296 | 0.240 | 0.240 | 0.200 | 6.240 |
| textgrad_ppo_slm | 0.493 | 0.533 | 0.360 | 0.120 | 0.200 | 6.160 |

## Per Environment

| method | env | reward | score | success | invalid | truncated | turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | GuessTheNumber-v0 | 0.684 | 0.588 | 0.400 | 0.400 | 0.200 | 9.200 |
| fixed_prompt_slm | FrozenLake-v0 | 0.467 | 0.249 | 0.000 | 0.400 | 0.000 | 3.600 |
| fixed_prompt_slm | Blackjack-v0 | 0.400 | 0.375 | 0.000 | 0.000 | 0.000 | 5.000 |
| fixed_prompt_slm | Nim-v0 | 0.200 | 0.269 | 0.600 | 0.400 | 0.000 | 6.200 |
| fixed_prompt_slm | Mastermind-v0 | 0.150 | -0.258 | 0.000 | 0.400 | 0.600 | 11.600 |
| textgrad_policy_iteration_slm | GuessTheNumber-v0 | 0.884 | 0.958 | 0.600 | 0.400 | 0.000 | 5.200 |
| textgrad_policy_iteration_slm | FrozenLake-v0 | 0.400 | 0.175 | 0.000 | 0.400 | 0.000 | 5.000 |
| textgrad_policy_iteration_slm | Blackjack-v0 | 0.400 | 0.374 | 0.000 | 0.000 | 0.000 | 5.200 |
| textgrad_policy_iteration_slm | Nim-v0 | 0.200 | 0.281 | 0.600 | 0.400 | 0.000 | 3.800 |
| textgrad_policy_iteration_slm | Mastermind-v0 | 0.000 | -0.310 | 0.000 | 0.000 | 1.000 | 12.000 |
| textgrad_ppo_slm | GuessTheNumber-v0 | 1.000 | 1.473 | 1.000 | 0.000 | 0.000 | 5.400 |
| textgrad_ppo_slm | FrozenLake-v0 | 0.467 | 0.249 | 0.000 | 0.400 | 0.000 | 3.600 |
| textgrad_ppo_slm | Blackjack-v0 | 0.400 | 0.375 | 0.000 | 0.000 | 0.000 | 5.000 |
| textgrad_ppo_slm | Nim-v0 | 0.600 | 0.876 | 0.800 | 0.200 | 0.000 | 4.800 |
| textgrad_ppo_slm | Mastermind-v0 | 0.000 | -0.310 | 0.000 | 0.000 | 1.000 | 12.000 |
