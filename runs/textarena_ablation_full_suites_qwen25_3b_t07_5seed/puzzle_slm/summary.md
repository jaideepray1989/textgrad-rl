# TextArena puzzle SLM Suite

- Model: qwen2.5:3b
- Temperature: 0.7

| method | reward | score | success | invalid | truncated | turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| textgrad_rl_no_gate_slm | 0.089 | -0.427 | 0.000 | 1.000 | 0.000 | 3.250 |
| textgrad_rl_train_val_slm | 0.109 | -0.406 | 0.000 | 1.000 | 0.000 | 3.100 |
| textgrad_rl_kl_gate_slm | 0.097 | -0.417 | 0.000 | 1.000 | 0.000 | 2.900 |
| textgrad_rl_clipped_surrogate_slm | 0.075 | -0.416 | 0.000 | 0.950 | 0.000 | 3.150 |
| textgrad_rl_ppo_slm | 0.129 | -0.387 | 0.000 | 1.000 | 0.000 | 3.300 |

## Per Environment

| method | env | reward | score | success | invalid | truncated | turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| textgrad_rl_no_gate_slm | Wordle-v0 | 0.220 | -0.301 | 0.000 | 1.000 | 0.000 | 4.200 |
| textgrad_rl_no_gate_slm | Hangman-v0 | 0.080 | -0.435 | 0.000 | 1.000 | 0.000 | 3.000 |
| textgrad_rl_no_gate_slm | Minesweeper-v0-small | 0.057 | -0.462 | 0.000 | 1.000 | 0.000 | 3.800 |
| textgrad_rl_no_gate_slm | Sudoku-v0-very-easy | 0.000 | -0.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| textgrad_rl_train_val_slm | Wordle-v0 | 0.300 | -0.219 | 0.000 | 1.000 | 0.000 | 3.800 |
| textgrad_rl_train_val_slm | Hangman-v0 | 0.080 | -0.434 | 0.000 | 1.000 | 0.000 | 2.800 |
| textgrad_rl_train_val_slm | Minesweeper-v0-small | 0.057 | -0.462 | 0.000 | 1.000 | 0.000 | 3.800 |
| textgrad_rl_train_val_slm | Sudoku-v0-very-easy | 0.000 | -0.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| textgrad_rl_kl_gate_slm | Wordle-v0 | 0.280 | -0.237 | 0.000 | 1.000 | 0.000 | 3.400 |
| textgrad_rl_kl_gate_slm | Hangman-v0 | 0.080 | -0.434 | 0.000 | 1.000 | 0.000 | 2.800 |
| textgrad_rl_kl_gate_slm | Minesweeper-v0-small | 0.029 | -0.488 | 0.000 | 1.000 | 0.000 | 3.400 |
| textgrad_rl_kl_gate_slm | Sudoku-v0-very-easy | 0.000 | -0.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| textgrad_rl_clipped_surrogate_slm | Wordle-v0 | 0.240 | -0.183 | 0.000 | 0.800 | 0.000 | 4.600 |
| textgrad_rl_clipped_surrogate_slm | Hangman-v0 | 0.040 | -0.473 | 0.000 | 1.000 | 0.000 | 2.600 |
| textgrad_rl_clipped_surrogate_slm | Minesweeper-v0-small | 0.018 | -0.499 | 0.000 | 1.000 | 0.000 | 3.400 |
| textgrad_rl_clipped_surrogate_slm | Sudoku-v0-very-easy | 0.000 | -0.510 | 0.000 | 1.000 | 0.000 | 2.000 |
| textgrad_rl_ppo_slm | Wordle-v0 | 0.380 | -0.144 | 0.000 | 1.000 | 0.000 | 4.800 |
| textgrad_rl_ppo_slm | Hangman-v0 | 0.080 | -0.435 | 0.000 | 1.000 | 0.000 | 3.000 |
| textgrad_rl_ppo_slm | Minesweeper-v0-small | 0.057 | -0.460 | 0.000 | 1.000 | 0.000 | 3.400 |
| textgrad_rl_ppo_slm | Sudoku-v0-very-easy | 0.000 | -0.510 | 0.000 | 1.000 | 0.000 | 2.000 |
