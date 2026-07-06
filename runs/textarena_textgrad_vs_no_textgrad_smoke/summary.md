# TextArena TextGrad-RL vs No TextGrad

- Environment: `TicTacToe-v0`
- Opponents: first_available, center_first, random_legal, optimal_minimax
- Test games per variant: 16

## Held-Out Test Results

| variant | games | wins | draws | losses | win_rate | non_loss_rate | avg_reward | invalid_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_textgrad | 16 | 7 | 4 | 5 | 0.438 | 0.688 | 0.125 | 0.000 |
| textgrad_rl | 16 | 12 | 4 | 0 | 0.750 | 1.000 | 0.750 | 0.000 |

## Delta

- Win rate: +0.312
- Non-loss rate: +0.312
- Average reward: +0.625

## Final Text Variables

### game_strategy_prompt

```text
Play legal TicTacToe moves. Prefer the center when available. Prefer corners before edges.

Learned rules:
- block opponent immediate winning moves before center or corner preferences.
```
