# TextArena Benchmark Summary

- Environment: `TicTacToe-v0`
- Games: 16
- Invalid-move games: 0

## Scoreboard

| agent | games | wins | draws | losses | win_rate | non_loss_rate | avg_reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| optimal_minimax | 8 | 6 | 2 | 0 | 0.750 | 1.000 | 0.750 |
| center_first | 8 | 4 | 2 | 2 | 0.500 | 0.750 | 0.250 |
| random_legal | 8 | 2 | 1 | 5 | 0.250 | 0.375 | -0.375 |
| first_available | 8 | 1 | 1 | 6 | 0.125 | 0.250 | -0.625 |

## Matchups

| player0 | player1 | games | p0_wins | p1_wins | draws | p0_avg_reward |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| center_first | center_first | 1 | 0 | 0 | 1 | 0.000 |
| center_first | first_available | 1 | 1 | 0 | 0 | 1.000 |
| center_first | optimal_minimax | 1 | 0 | 1 | 0 | -1.000 |
| center_first | random_legal | 1 | 1 | 0 | 0 | 1.000 |
| first_available | center_first | 1 | 0 | 1 | 0 | -1.000 |
| first_available | first_available | 1 | 1 | 0 | 0 | 1.000 |
| first_available | optimal_minimax | 1 | 0 | 1 | 0 | -1.000 |
| first_available | random_legal | 1 | 0 | 0 | 1 | 0.000 |
| optimal_minimax | center_first | 1 | 1 | 0 | 0 | 1.000 |
| optimal_minimax | first_available | 1 | 1 | 0 | 0 | 1.000 |
| optimal_minimax | optimal_minimax | 1 | 0 | 0 | 1 | 0.000 |
| optimal_minimax | random_legal | 1 | 1 | 0 | 0 | 1.000 |
| random_legal | center_first | 1 | 0 | 1 | 0 | -1.000 |
| random_legal | first_available | 1 | 1 | 0 | 0 | 1.000 |
| random_legal | optimal_minimax | 1 | 0 | 1 | 0 | -1.000 |
| random_legal | random_legal | 1 | 1 | 0 | 0 | 1.000 |
