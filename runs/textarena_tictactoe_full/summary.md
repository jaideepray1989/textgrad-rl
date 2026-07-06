# TextArena Benchmark Summary

- Environment: `TicTacToe-v0`
- Games: 320
- Invalid-move games: 0

## Scoreboard

| agent | games | wins | draws | losses | win_rate | non_loss_rate | avg_reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| optimal_minimax | 160 | 115 | 45 | 0 | 0.719 | 1.000 | 0.719 |
| center_first | 160 | 71 | 40 | 49 | 0.444 | 0.694 | 0.138 |
| first_available | 160 | 46 | 1 | 113 | 0.287 | 0.294 | -0.419 |
| random_legal | 160 | 38 | 14 | 108 | 0.237 | 0.325 | -0.438 |

## Matchups

| player0 | player1 | games | p0_wins | p1_wins | draws | p0_avg_reward |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| center_first | center_first | 20 | 0 | 0 | 20 | 0.000 |
| center_first | first_available | 20 | 20 | 0 | 0 | 1.000 |
| center_first | optimal_minimax | 20 | 0 | 20 | 0 | -1.000 |
| center_first | random_legal | 20 | 18 | 2 | 0 | 0.800 |
| first_available | center_first | 20 | 0 | 20 | 0 | -1.000 |
| first_available | first_available | 20 | 20 | 0 | 0 | 1.000 |
| first_available | optimal_minimax | 20 | 0 | 20 | 0 | -1.000 |
| first_available | random_legal | 20 | 19 | 0 | 1 | 0.950 |
| optimal_minimax | center_first | 20 | 20 | 0 | 0 | 1.000 |
| optimal_minimax | first_available | 20 | 20 | 0 | 0 | 1.000 |
| optimal_minimax | optimal_minimax | 20 | 0 | 0 | 20 | 0.000 |
| optimal_minimax | random_legal | 20 | 19 | 0 | 1 | 0.950 |
| random_legal | center_first | 20 | 7 | 13 | 0 | -0.300 |
| random_legal | first_available | 20 | 13 | 7 | 0 | 0.300 |
| random_legal | optimal_minimax | 20 | 0 | 16 | 4 | -0.800 |
| random_legal | random_legal | 20 | 10 | 6 | 4 | 0.200 |
