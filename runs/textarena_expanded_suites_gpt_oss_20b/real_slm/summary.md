# TextArena real_slm SLM Suite

- Model: gpt-oss:20b

| method | reward | score | success | invalid | truncated | turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | -0.084 | -0.598 | 0.000 | 1.000 | 0.000 | 2.800 |
| scalar_prompt_search_slm | -0.084 | -0.597 | 0.000 | 1.000 | 0.000 | 2.600 |
| modular_textgrad_slm | -0.084 | -0.598 | 0.000 | 1.000 | 0.000 | 2.800 |
| textgrad_rl_plus_slm | -0.084 | -0.598 | 0.000 | 1.000 | 0.000 | 2.800 |
| textgrad_policy_iteration_slm | -0.084 | -0.598 | 0.000 | 1.000 | 0.000 | 2.800 |

## Per Environment

| method | env | reward | score | success | invalid | truncated | turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | GuessTheNumber-v0 | 0.579 | 0.064 | 0.000 | 1.000 | 0.000 | 3.000 |
| fixed_prompt_slm | FrozenLake-v0 | 0.000 | -0.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| fixed_prompt_slm | Blackjack-v0 | 0.000 | -0.525 | 0.000 | 1.000 | 0.000 | 5.000 |
| fixed_prompt_slm | Nim-v0 | -1.000 | -1.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| fixed_prompt_slm | Mastermind-v0 | 0.000 | -0.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| scalar_prompt_search_slm | GuessTheNumber-v0 | 0.579 | 0.064 | 0.000 | 1.000 | 0.000 | 3.000 |
| scalar_prompt_search_slm | FrozenLake-v0 | 0.000 | -0.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| scalar_prompt_search_slm | Blackjack-v0 | 0.000 | -0.520 | 0.000 | 1.000 | 0.000 | 4.000 |
| scalar_prompt_search_slm | Nim-v0 | -1.000 | -1.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| scalar_prompt_search_slm | Mastermind-v0 | 0.000 | -0.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| modular_textgrad_slm | GuessTheNumber-v0 | 0.579 | 0.064 | 0.000 | 1.000 | 0.000 | 3.000 |
| modular_textgrad_slm | FrozenLake-v0 | 0.000 | -0.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| modular_textgrad_slm | Blackjack-v0 | 0.000 | -0.525 | 0.000 | 1.000 | 0.000 | 5.000 |
| modular_textgrad_slm | Nim-v0 | -1.000 | -1.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| modular_textgrad_slm | Mastermind-v0 | 0.000 | -0.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| textgrad_rl_plus_slm | GuessTheNumber-v0 | 0.579 | 0.064 | 0.000 | 1.000 | 0.000 | 3.000 |
| textgrad_rl_plus_slm | FrozenLake-v0 | 0.000 | -0.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| textgrad_rl_plus_slm | Blackjack-v0 | 0.000 | -0.525 | 0.000 | 1.000 | 0.000 | 5.000 |
| textgrad_rl_plus_slm | Nim-v0 | -1.000 | -1.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| textgrad_rl_plus_slm | Mastermind-v0 | 0.000 | -0.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| textgrad_policy_iteration_slm | GuessTheNumber-v0 | 0.579 | 0.064 | 0.000 | 1.000 | 0.000 | 3.000 |
| textgrad_policy_iteration_slm | FrozenLake-v0 | 0.000 | -0.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| textgrad_policy_iteration_slm | Blackjack-v0 | 0.000 | -0.525 | 0.000 | 1.000 | 0.000 | 5.000 |
| textgrad_policy_iteration_slm | Nim-v0 | -1.000 | -1.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| textgrad_policy_iteration_slm | Mastermind-v0 | 0.000 | -0.510 | 0.000 | 1.000 | 0.000 | 2.000 |
