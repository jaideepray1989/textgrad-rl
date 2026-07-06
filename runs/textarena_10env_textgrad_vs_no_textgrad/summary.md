# TextArena Multi-Environment Comparison

- Environments: 10
- Problems: Nim-v0, ConnectFour-v0, ReverseTicTacToe-v0, GuessTheNumber-v0, FrozenLake-v0, TowerOfHanoi-v0, LightsOut-v0, Mastermind-v0, Blackjack-v0, Bandit-v0

## Overall

| variant | episodes | avg_reward | success_rate | invalid_rate | avg_turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_textgrad | 130 | 0.162 | 0.215 | 0.115 | 12.077 |
| textgrad_rl | 130 | 0.557 | 0.615 | 0.000 | 10.800 |

## Delta

- Average reward: +0.394
- Success rate: +0.400
- Invalid move rate: -0.115

## Per Environment

| env | no_textgrad_reward | textgrad_reward | delta_reward | no_textgrad_success | textgrad_success |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nim-v0 | 0.000 | 1.000 | +1.000 | 0.500 | 1.000 |
| ConnectFour-v0 | 0.000 | 0.000 | +0.000 | 0.500 | 0.500 |
| ReverseTicTacToe-v0 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 |
| GuessTheNumber-v0 | 0.974 | 1.000 | +0.026 | 0.800 | 1.000 |
| FrozenLake-v0 | 0.467 | 1.000 | +0.533 | 0.000 | 1.000 |
| TowerOfHanoi-v0 | 0.000 | 1.000 | +1.000 | 0.000 | 1.000 |
| LightsOut-v0 | -0.201 | 1.000 | +1.201 | 0.000 | 1.000 |
| Mastermind-v0 | 0.412 | 1.000 | +0.588 | 0.000 | 1.000 |
| Blackjack-v0 | 0.280 | 0.060 | -0.220 | 0.000 | 0.000 |
| Bandit-v0 | 0.179 | 0.179 | +0.000 | 0.000 | 0.000 |

## Final Text Rules

```text
Play valid TextArena actions. Prefer simple legal moves, center columns/cells, standing in Blackjack, and first available puzzle moves.

Learned rules:
- For Bandit, explore every button once, track empirical reward, then exploit the best observed button.
- For Blackjack, use the basic threshold policy: hit below 17 and stand at 17 or higher.
- For Connect Four, take an immediate winning column and otherwise block an opponent immediate connect-four.
- For Frozen Lake, plan a shortest safe path to the goal while avoiding holes.
- For Guess The Number, maintain lower and upper bounds from higher/lower hints and guess the midpoint.
- For Lights Out, solve the binary toggle system and press cells from a valid solution.
- For Mastermind, maintain all candidate codes and eliminate candidates inconsistent with black/white feedback.
- For Nim, use the xor strategy: move to a zero nim-sum when possible.
- For Reverse Tic Tac Toe, avoid moves that immediately complete your own three-in-a-row.
- For Tower of Hanoi, follow the recursive optimal disk-moving plan from A to C.
```
