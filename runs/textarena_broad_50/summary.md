# TextArena Broad 50-Environment Suite

Environments: 50
Test seeds: 3
Turn budget: 80
Supported policy-family envs: 19
Generic fallback envs: 31

## Overall Results

| Method | Envs | Episodes | Reward | Success | Invalid | Repeated | Turns | Updates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_prompt | 50 | 195 | 0.094 | 0.185 | 0.518 | 0.749 | 9.32 | 0 |
| textgrad_policy_iteration | 50 | 195 | 0.261 | 0.369 | 0.451 | 0.718 | 8.50 | 6 |

## Supported vs Fallback

| Method | Slice | Envs | Episodes | Reward | Success | Invalid | Repeated | Turns |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_prompt | generic_fallback | 31 | 123 | 0.048 | 0.195 | 0.691 | 0.699 | 7.00 |
| fixed_prompt | supported_policy_family | 19 | 72 | 0.173 | 0.167 | 0.222 | 0.833 | 13.29 |
| textgrad_policy_iteration | generic_fallback | 31 | 123 | 0.048 | 0.195 | 0.691 | 0.699 | 7.00 |
| textgrad_policy_iteration | supported_policy_family | 19 | 72 | 0.626 | 0.667 | 0.042 | 0.750 | 11.07 |

## Category Results

| Method | Category | Envs | Episodes | Reward | Success | Invalid | Repeated | Turns |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_prompt | board_game | 10 | 60 | 0.000 | 0.450 | 0.300 | 0.700 | 7.80 |
| fixed_prompt | card_game | 2 | 12 | 0.000 | 0.500 | 0.500 | 1.000 | 3.83 |
| fixed_prompt | deduction | 4 | 12 | 0.592 | 0.000 | 0.500 | 0.500 | 8.00 |
| fixed_prompt | language_puzzle | 6 | 18 | 0.240 | 0.000 | 0.889 | 0.722 | 6.17 |
| fixed_prompt | planning | 6 | 18 | 0.194 | 0.000 | 0.556 | 0.889 | 10.00 |
| fixed_prompt | puzzle | 9 | 27 | 0.012 | 0.000 | 0.889 | 0.889 | 7.52 |
| fixed_prompt | social_game | 3 | 18 | 0.000 | 0.167 | 0.167 | 1.000 | 24.67 |
| fixed_prompt | stochastic_decision | 6 | 18 | 0.170 | 0.000 | 0.333 | 0.667 | 13.67 |
| fixed_prompt | symbolic_puzzle | 4 | 12 | 0.000 | 0.000 | 1.000 | 0.250 | 2.00 |
| textgrad_policy_iteration | board_game | 10 | 60 | 0.200 | 0.550 | 0.300 | 0.700 | 7.20 |
| textgrad_policy_iteration | card_game | 2 | 12 | 0.000 | 0.500 | 0.500 | 1.000 | 3.83 |
| textgrad_policy_iteration | deduction | 4 | 12 | 1.000 | 1.000 | 0.000 | 0.250 | 4.33 |
| textgrad_policy_iteration | language_puzzle | 6 | 18 | 0.240 | 0.000 | 0.889 | 0.722 | 6.17 |
| textgrad_policy_iteration | planning | 6 | 18 | 0.875 | 0.833 | 0.167 | 0.889 | 7.50 |
| textgrad_policy_iteration | puzzle | 9 | 27 | 0.140 | 0.111 | 0.889 | 0.778 | 6.22 |
| textgrad_policy_iteration | social_game | 3 | 18 | 0.000 | 0.167 | 0.167 | 1.000 | 24.67 |
| textgrad_policy_iteration | stochastic_decision | 6 | 18 | 0.170 | 0.000 | 0.333 | 0.667 | 13.67 |
| textgrad_policy_iteration | symbolic_puzzle | 4 | 12 | 0.000 | 0.000 | 1.000 | 0.250 | 2.00 |
