# TextArena Real SLM Comparison

- Environment: GuessTheNumber-v0
- Model: qwen2.5:3b
- Base URL: http://localhost:11434/v1
- TextGrad update accepted: False

## Held-Out Test

| variant | episodes | avg_reward | success | invalid | avg_turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_textgrad | 1 | 0.526 | 0.000 | 0.000 | 11.000 |
| textgrad_rl | 1 | 0.526 | 0.000 | 0.000 | 11.000 |

## Delta

- Average reward: +0.000
- Success rate: +0.000
- Invalid move rate: +0.000

## Textual Gradients

- No gradient emitted; training episodes were already clean.
