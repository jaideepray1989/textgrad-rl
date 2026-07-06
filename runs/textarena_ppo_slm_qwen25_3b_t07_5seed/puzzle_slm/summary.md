# TextArena puzzle SLM Suite

- Model: qwen2.5:3b
- Temperature: 0.7

| method | reward | score | success | invalid | truncated | turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | 0.107 | -0.409 | 0.000 | 1.000 | 0.000 | 3.150 |
| textgrad_policy_iteration_slm | 0.062 | -0.454 | 0.000 | 1.000 | 0.000 | 3.300 |
| textgrad_ppo_slm | 0.054 | -0.461 | 0.000 | 1.000 | 0.000 | 3.000 |

## Per Environment

| method | env | reward | score | success | invalid | truncated | turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | Wordle-v0 | 0.320 | -0.201 | 0.000 | 1.000 | 0.000 | 4.200 |
| fixed_prompt_slm | Hangman-v0 | 0.080 | -0.435 | 0.000 | 1.000 | 0.000 | 3.000 |
| fixed_prompt_slm | Minesweeper-v0-small | 0.029 | -0.488 | 0.000 | 1.000 | 0.000 | 3.400 |
| fixed_prompt_slm | Sudoku-v0-very-easy | 0.000 | -0.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| textgrad_policy_iteration_slm | Wordle-v0 | 0.140 | -0.385 | 0.000 | 1.000 | 0.000 | 5.000 |
| textgrad_policy_iteration_slm | Hangman-v0 | 0.080 | -0.434 | 0.000 | 1.000 | 0.000 | 2.800 |
| textgrad_policy_iteration_slm | Minesweeper-v0-small | 0.029 | -0.488 | 0.000 | 1.000 | 0.000 | 3.400 |
| textgrad_policy_iteration_slm | Sudoku-v0-very-easy | 0.000 | -0.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| textgrad_ppo_slm | Wordle-v0 | 0.160 | -0.360 | 0.000 | 1.000 | 0.000 | 4.000 |
| textgrad_ppo_slm | Hangman-v0 | 0.040 | -0.473 | 0.000 | 1.000 | 0.000 | 2.600 |
| textgrad_ppo_slm | Minesweeper-v0-small | 0.014 | -0.503 | 0.000 | 1.000 | 0.000 | 3.400 |
| textgrad_ppo_slm | Sudoku-v0-very-easy | 0.000 | -0.510 | 0.000 | 1.000 | 0.000 | 2.000 |
