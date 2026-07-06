# TextArena Difficulty Generalization

| method | reward | success | invalid | turns |
| --- | ---: | ---: | ---: | ---: |
| fixed_prompt | 0.190 | 0.167 | 0.139 | 22.597 |
| modular_textgrad | 0.830 | 0.792 | 0.000 | 18.236 |
| textgrad_rl_plus | 0.880 | 0.792 | 0.000 | 17.250 |
| textgrad_policy_iteration | 0.897 | 0.833 | 0.000 | 16.958 |

## Per Environment

| method | env | reward | success | invalid | turns |
| --- | --- | ---: | ---: | ---: | ---: |
| fixed_prompt | GuessTheNumber-v0-hardcore | 0.561 | 0.000 | 0.000 | 11.000 |
| fixed_prompt | FrozenLake-v0-random | 0.333 | 0.000 | 0.667 | 3.333 |
| fixed_prompt | FrozenLake-v0-hardcore | 0.229 | 0.000 | 0.000 | 1.833 |
| fixed_prompt | Nim-v0-medium | 0.000 | 0.500 | 0.000 | 16.000 |
| fixed_prompt | Nim-v0-large | 0.000 | 0.500 | 0.000 | 34.000 |
| fixed_prompt | TowerOfHanoi-v0-medium | 0.000 | 0.000 | 0.000 | 31.000 |
| fixed_prompt | TowerOfHanoi-v0-hard | 0.000 | 0.000 | 0.000 | 63.000 |
| fixed_prompt | Blackjack-v0-long | 0.500 | 0.000 | 0.000 | 15.000 |
| fixed_prompt | Bandit-v0-hard | 0.263 | 0.000 | 0.000 | 41.000 |
| fixed_prompt | Mastermind-v0-hard | 0.396 | 0.000 | 1.000 | 5.000 |
| modular_textgrad | GuessTheNumber-v0-hardcore | 0.796 | 0.500 | 0.000 | 8.167 |
| modular_textgrad | FrozenLake-v0-random | 1.000 | 1.000 | 0.000 | 6.000 |
| modular_textgrad | FrozenLake-v0-hardcore | 1.000 | 1.000 | 0.000 | 8.000 |
| modular_textgrad | Nim-v0-medium | 1.000 | 1.000 | 0.000 | 10.500 |
| modular_textgrad | Nim-v0-large | 1.000 | 1.000 | 0.000 | 28.500 |
| modular_textgrad | TowerOfHanoi-v0-medium | 1.000 | 1.000 | 0.000 | 15.000 |
| modular_textgrad | TowerOfHanoi-v0-hard | 1.000 | 1.000 | 0.000 | 31.000 |
| modular_textgrad | Blackjack-v0-long | 0.056 | 0.000 | 0.000 | 26.833 |
| modular_textgrad | Bandit-v0-hard | 0.106 | 0.000 | 0.000 | 41.000 |
| modular_textgrad | Mastermind-v0-hard | 1.000 | 1.000 | 0.000 | 4.833 |
| textgrad_rl_plus | GuessTheNumber-v0-hardcore | 0.796 | 0.500 | 0.000 | 8.167 |
| textgrad_rl_plus | FrozenLake-v0-random | 1.000 | 1.000 | 0.000 | 6.000 |
| textgrad_rl_plus | FrozenLake-v0-hardcore | 1.000 | 1.000 | 0.000 | 8.000 |
| textgrad_rl_plus | Nim-v0-medium | 1.000 | 1.000 | 0.000 | 10.500 |
| textgrad_rl_plus | Nim-v0-large | 1.000 | 1.000 | 0.000 | 28.500 |
| textgrad_rl_plus | TowerOfHanoi-v0-medium | 1.000 | 1.000 | 0.000 | 15.000 |
| textgrad_rl_plus | TowerOfHanoi-v0-hard | 1.000 | 1.000 | 0.000 | 31.000 |
| textgrad_rl_plus | Blackjack-v0-long | 0.500 | 0.000 | 0.000 | 15.000 |
| textgrad_rl_plus | Bandit-v0-hard | 0.263 | 0.000 | 0.000 | 41.000 |
| textgrad_rl_plus | Mastermind-v0-hard | 1.000 | 1.000 | 0.000 | 4.833 |
| textgrad_policy_iteration | GuessTheNumber-v0-hardcore | 1.000 | 1.000 | 0.000 | 4.667 |
| textgrad_policy_iteration | FrozenLake-v0-random | 1.000 | 1.000 | 0.000 | 6.000 |
| textgrad_policy_iteration | FrozenLake-v0-hardcore | 1.000 | 1.000 | 0.000 | 8.000 |
| textgrad_policy_iteration | Nim-v0-medium | 1.000 | 1.000 | 0.000 | 10.500 |
| textgrad_policy_iteration | Nim-v0-large | 1.000 | 1.000 | 0.000 | 28.500 |
| textgrad_policy_iteration | TowerOfHanoi-v0-medium | 1.000 | 1.000 | 0.000 | 15.000 |
| textgrad_policy_iteration | TowerOfHanoi-v0-hard | 1.000 | 1.000 | 0.000 | 31.000 |
| textgrad_policy_iteration | Blackjack-v0-long | 0.500 | 0.000 | 0.000 | 15.000 |
| textgrad_policy_iteration | Bandit-v0-hard | 0.263 | 0.000 | 0.000 | 41.000 |
| textgrad_policy_iteration | Mastermind-v0-hard | 1.000 | 1.000 | 0.000 | 4.833 |
