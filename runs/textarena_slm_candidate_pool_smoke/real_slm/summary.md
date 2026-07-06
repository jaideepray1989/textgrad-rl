# TextArena real_slm SLM Suite

- Model: qwen2.5:3b
- Temperature: 0.7

| method | reward | score | success | invalid | truncated | turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| textgrad_rl_ppo_slm | 0.380 | 0.343 | 0.200 | 0.000 | 0.400 | 7.400 |

## Per Environment

| method | env | reward | score | success | invalid | truncated | turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| textgrad_rl_ppo_slm | GuessTheNumber-v0 | 0.000 | -0.310 | 0.000 | 0.000 | 1.000 | 12.000 |
| textgrad_rl_ppo_slm | FrozenLake-v0 | 0.500 | 0.485 | 0.000 | 0.000 | 0.000 | 3.000 |
| textgrad_rl_ppo_slm | Blackjack-v0 | 0.400 | 0.375 | 0.000 | 0.000 | 0.000 | 5.000 |
| textgrad_rl_ppo_slm | Nim-v0 | 1.000 | 1.475 | 1.000 | 0.000 | 0.000 | 5.000 |
| textgrad_rl_ppo_slm | Mastermind-v0 | 0.000 | -0.310 | 0.000 | 0.000 | 1.000 | 12.000 |
