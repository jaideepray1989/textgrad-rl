# TextArena puzzle SLM Suite

- Model: qwen2.5:3b
- Temperature: 0.7

| method | reward | score | success | invalid | truncated | turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | 0.146 | -0.372 | 0.000 | 1.000 | 0.000 | 3.750 |
| textgrad_policy_iteration_slm | 0.121 | -0.396 | 0.000 | 1.000 | 0.000 | 3.500 |
| textgrad_ppo_slm | 0.146 | -0.375 | 0.000 | 1.000 | 0.000 | 4.250 |

## Per Environment

| method | env | reward | score | success | invalid | truncated | turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | Wordle-v0 | 0.100 | -0.415 | 0.000 | 1.000 | 0.000 | 3.000 |
| fixed_prompt_slm | Hangman-v0 | 0.200 | -0.315 | 0.000 | 1.000 | 0.000 | 3.000 |
| fixed_prompt_slm | Minesweeper-v0-small | 0.286 | -0.249 | 0.000 | 1.000 | 0.000 | 7.000 |
| fixed_prompt_slm | Sudoku-v0-very-easy | 0.000 | -0.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| textgrad_policy_iteration_slm | Wordle-v0 | 0.000 | -0.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| textgrad_policy_iteration_slm | Hangman-v0 | 0.200 | -0.315 | 0.000 | 1.000 | 0.000 | 3.000 |
| textgrad_policy_iteration_slm | Minesweeper-v0-small | 0.286 | -0.249 | 0.000 | 1.000 | 0.000 | 7.000 |
| textgrad_policy_iteration_slm | Sudoku-v0-very-easy | 0.000 | -0.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| textgrad_ppo_slm | Wordle-v0 | 0.100 | -0.425 | 0.000 | 1.000 | 0.000 | 5.000 |
| textgrad_ppo_slm | Hangman-v0 | 0.200 | -0.315 | 0.000 | 1.000 | 0.000 | 3.000 |
| textgrad_ppo_slm | Minesweeper-v0-small | 0.286 | -0.249 | 0.000 | 1.000 | 0.000 | 7.000 |
| textgrad_ppo_slm | Sudoku-v0-very-easy | 0.000 | -0.510 | 0.000 | 1.000 | 0.000 | 2.000 |
