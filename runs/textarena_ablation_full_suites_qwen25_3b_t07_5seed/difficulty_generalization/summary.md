# TextArena Difficulty Generalization

| method | reward | success | invalid | turns |
| --- | ---: | ---: | ---: | ---: |
| textgrad_rl_no_gate | 0.860 | 0.833 | 0.000 | 17.944 |
| textgrad_rl_train_val | 0.897 | 0.833 | 0.000 | 16.958 |
| textgrad_rl_kl_gate | 0.897 | 0.833 | 0.000 | 16.958 |
| textgrad_rl_clipped_surrogate | 0.897 | 0.833 | 0.000 | 16.958 |
| textgrad_rl_ppo | 0.897 | 0.833 | 0.000 | 16.958 |

## Per Environment

| method | env | reward | success | invalid | turns |
| --- | --- | ---: | ---: | ---: | ---: |
| textgrad_rl_no_gate | GuessTheNumber-v0-hardcore | 1.000 | 1.000 | 0.000 | 4.667 |
| textgrad_rl_no_gate | FrozenLake-v0-random | 1.000 | 1.000 | 0.000 | 6.000 |
| textgrad_rl_no_gate | FrozenLake-v0-hardcore | 1.000 | 1.000 | 0.000 | 8.000 |
| textgrad_rl_no_gate | Nim-v0-medium | 1.000 | 1.000 | 0.000 | 10.500 |
| textgrad_rl_no_gate | Nim-v0-large | 1.000 | 1.000 | 0.000 | 28.500 |
| textgrad_rl_no_gate | TowerOfHanoi-v0-medium | 1.000 | 1.000 | 0.000 | 15.000 |
| textgrad_rl_no_gate | TowerOfHanoi-v0-hard | 1.000 | 1.000 | 0.000 | 31.000 |
| textgrad_rl_no_gate | Blackjack-v0-long | 0.056 | 0.000 | 0.000 | 26.833 |
| textgrad_rl_no_gate | Bandit-v0-hard | 0.263 | 0.000 | 0.000 | 41.000 |
| textgrad_rl_no_gate | Mastermind-v0-hard | 1.000 | 1.000 | 0.000 | 4.833 |
| textgrad_rl_train_val | GuessTheNumber-v0-hardcore | 1.000 | 1.000 | 0.000 | 4.667 |
| textgrad_rl_train_val | FrozenLake-v0-random | 1.000 | 1.000 | 0.000 | 6.000 |
| textgrad_rl_train_val | FrozenLake-v0-hardcore | 1.000 | 1.000 | 0.000 | 8.000 |
| textgrad_rl_train_val | Nim-v0-medium | 1.000 | 1.000 | 0.000 | 10.500 |
| textgrad_rl_train_val | Nim-v0-large | 1.000 | 1.000 | 0.000 | 28.500 |
| textgrad_rl_train_val | TowerOfHanoi-v0-medium | 1.000 | 1.000 | 0.000 | 15.000 |
| textgrad_rl_train_val | TowerOfHanoi-v0-hard | 1.000 | 1.000 | 0.000 | 31.000 |
| textgrad_rl_train_val | Blackjack-v0-long | 0.500 | 0.000 | 0.000 | 15.000 |
| textgrad_rl_train_val | Bandit-v0-hard | 0.263 | 0.000 | 0.000 | 41.000 |
| textgrad_rl_train_val | Mastermind-v0-hard | 1.000 | 1.000 | 0.000 | 4.833 |
| textgrad_rl_kl_gate | GuessTheNumber-v0-hardcore | 1.000 | 1.000 | 0.000 | 4.667 |
| textgrad_rl_kl_gate | FrozenLake-v0-random | 1.000 | 1.000 | 0.000 | 6.000 |
| textgrad_rl_kl_gate | FrozenLake-v0-hardcore | 1.000 | 1.000 | 0.000 | 8.000 |
| textgrad_rl_kl_gate | Nim-v0-medium | 1.000 | 1.000 | 0.000 | 10.500 |
| textgrad_rl_kl_gate | Nim-v0-large | 1.000 | 1.000 | 0.000 | 28.500 |
| textgrad_rl_kl_gate | TowerOfHanoi-v0-medium | 1.000 | 1.000 | 0.000 | 15.000 |
| textgrad_rl_kl_gate | TowerOfHanoi-v0-hard | 1.000 | 1.000 | 0.000 | 31.000 |
| textgrad_rl_kl_gate | Blackjack-v0-long | 0.500 | 0.000 | 0.000 | 15.000 |
| textgrad_rl_kl_gate | Bandit-v0-hard | 0.263 | 0.000 | 0.000 | 41.000 |
| textgrad_rl_kl_gate | Mastermind-v0-hard | 1.000 | 1.000 | 0.000 | 4.833 |
| textgrad_rl_clipped_surrogate | GuessTheNumber-v0-hardcore | 1.000 | 1.000 | 0.000 | 4.667 |
| textgrad_rl_clipped_surrogate | FrozenLake-v0-random | 1.000 | 1.000 | 0.000 | 6.000 |
| textgrad_rl_clipped_surrogate | FrozenLake-v0-hardcore | 1.000 | 1.000 | 0.000 | 8.000 |
| textgrad_rl_clipped_surrogate | Nim-v0-medium | 1.000 | 1.000 | 0.000 | 10.500 |
| textgrad_rl_clipped_surrogate | Nim-v0-large | 1.000 | 1.000 | 0.000 | 28.500 |
| textgrad_rl_clipped_surrogate | TowerOfHanoi-v0-medium | 1.000 | 1.000 | 0.000 | 15.000 |
| textgrad_rl_clipped_surrogate | TowerOfHanoi-v0-hard | 1.000 | 1.000 | 0.000 | 31.000 |
| textgrad_rl_clipped_surrogate | Blackjack-v0-long | 0.500 | 0.000 | 0.000 | 15.000 |
| textgrad_rl_clipped_surrogate | Bandit-v0-hard | 0.263 | 0.000 | 0.000 | 41.000 |
| textgrad_rl_clipped_surrogate | Mastermind-v0-hard | 1.000 | 1.000 | 0.000 | 4.833 |
| textgrad_rl_ppo | GuessTheNumber-v0-hardcore | 1.000 | 1.000 | 0.000 | 4.667 |
| textgrad_rl_ppo | FrozenLake-v0-random | 1.000 | 1.000 | 0.000 | 6.000 |
| textgrad_rl_ppo | FrozenLake-v0-hardcore | 1.000 | 1.000 | 0.000 | 8.000 |
| textgrad_rl_ppo | Nim-v0-medium | 1.000 | 1.000 | 0.000 | 10.500 |
| textgrad_rl_ppo | Nim-v0-large | 1.000 | 1.000 | 0.000 | 28.500 |
| textgrad_rl_ppo | TowerOfHanoi-v0-medium | 1.000 | 1.000 | 0.000 | 15.000 |
| textgrad_rl_ppo | TowerOfHanoi-v0-hard | 1.000 | 1.000 | 0.000 | 31.000 |
| textgrad_rl_ppo | Blackjack-v0-long | 0.500 | 0.000 | 0.000 | 15.000 |
| textgrad_rl_ppo | Bandit-v0-hard | 0.263 | 0.000 | 0.000 | 41.000 |
| textgrad_rl_ppo | Mastermind-v0-hard | 1.000 | 1.000 | 0.000 | 4.833 |
