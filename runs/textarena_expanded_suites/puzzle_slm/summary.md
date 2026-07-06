# TextArena puzzle SLM Suite

- Model: qwen2.5:3b

| method | reward | score | success | invalid | truncated | turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | 0.146 | -0.374 | 0.000 | 1.000 | 0.000 | 4.000 |
| scalar_prompt_search_slm | 0.146 | -0.374 | 0.000 | 1.000 | 0.000 | 4.000 |
| modular_textgrad_slm | 0.111 | -0.407 | 0.000 | 1.000 | 0.000 | 3.500 |
| textgrad_rl_plus_slm | 0.146 | -0.374 | 0.000 | 1.000 | 0.000 | 4.000 |
| textgrad_policy_iteration_slm | 0.111 | -0.407 | 0.000 | 1.000 | 0.000 | 3.500 |

## Per Environment

| method | env | reward | score | success | invalid | truncated | turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | Wordle-v0 | 0.100 | -0.420 | 0.000 | 1.000 | 0.000 | 4.000 |
| fixed_prompt_slm | Hangman-v0 | 0.200 | -0.315 | 0.000 | 1.000 | 0.000 | 3.000 |
| fixed_prompt_slm | Minesweeper-v0-small | 0.286 | -0.249 | 0.000 | 1.000 | 0.000 | 7.000 |
| fixed_prompt_slm | Sudoku-v0-very-easy | 0.000 | -0.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| scalar_prompt_search_slm | Wordle-v0 | 0.100 | -0.420 | 0.000 | 1.000 | 0.000 | 4.000 |
| scalar_prompt_search_slm | Hangman-v0 | 0.200 | -0.315 | 0.000 | 1.000 | 0.000 | 3.000 |
| scalar_prompt_search_slm | Minesweeper-v0-small | 0.286 | -0.249 | 0.000 | 1.000 | 0.000 | 7.000 |
| scalar_prompt_search_slm | Sudoku-v0-very-easy | 0.000 | -0.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| modular_textgrad_slm | Wordle-v0 | 0.100 | -0.420 | 0.000 | 1.000 | 0.000 | 4.000 |
| modular_textgrad_slm | Hangman-v0 | 0.200 | -0.315 | 0.000 | 1.000 | 0.000 | 3.000 |
| modular_textgrad_slm | Minesweeper-v0-small | 0.143 | -0.382 | 0.000 | 1.000 | 0.000 | 5.000 |
| modular_textgrad_slm | Sudoku-v0-very-easy | 0.000 | -0.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| textgrad_rl_plus_slm | Wordle-v0 | 0.100 | -0.420 | 0.000 | 1.000 | 0.000 | 4.000 |
| textgrad_rl_plus_slm | Hangman-v0 | 0.200 | -0.315 | 0.000 | 1.000 | 0.000 | 3.000 |
| textgrad_rl_plus_slm | Minesweeper-v0-small | 0.286 | -0.249 | 0.000 | 1.000 | 0.000 | 7.000 |
| textgrad_rl_plus_slm | Sudoku-v0-very-easy | 0.000 | -0.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| textgrad_policy_iteration_slm | Wordle-v0 | 0.100 | -0.420 | 0.000 | 1.000 | 0.000 | 4.000 |
| textgrad_policy_iteration_slm | Hangman-v0 | 0.200 | -0.315 | 0.000 | 1.000 | 0.000 | 3.000 |
| textgrad_policy_iteration_slm | Minesweeper-v0-small | 0.143 | -0.382 | 0.000 | 1.000 | 0.000 | 5.000 |
| textgrad_policy_iteration_slm | Sudoku-v0-very-easy | 0.000 | -0.510 | 0.000 | 1.000 | 0.000 | 2.000 |
