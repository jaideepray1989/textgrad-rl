# TextArena TextGrad-RL vs No TextGrad

- Environment: `TicTacToe-v0`
- Opponents: first_available, center_first, random_legal, optimal_minimax
- Test games per variant: 160

## Held-Out Test Results

| variant | games | wins | draws | losses | win_rate | non_loss_rate | avg_reward | invalid_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_textgrad | 160 | 68 | 43 | 49 | 0.425 | 0.694 | 0.119 | 0.000 |
| textgrad_rl | 160 | 120 | 40 | 0 | 0.750 | 1.000 | 0.750 | 0.000 |

## Delta

- Win rate: +0.325
- Non-loss rate: +0.306
- Average reward: +0.631

## Final Text Variables

### game_strategy_prompt

```text
Play legal TicTacToe moves. Prefer the center when available. Prefer corners before edges.

Learned rules:
- block opponent immediate winning moves before center or corner preferences.
```
