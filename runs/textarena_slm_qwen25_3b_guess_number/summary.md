# TextArena Real SLM Comparison

- Environment: GuessTheNumber-v0
- Model: qwen2.5:3b
- Base URL: http://localhost:11434/v1
- TextGrad update accepted: True

## Held-Out Test

| variant | episodes | avg_reward | success | invalid | avg_turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_textgrad | 5 | 0.905 | 0.800 | 0.000 | 6.000 |
| textgrad_rl | 5 | 0.916 | 0.800 | 0.000 | 5.800 |

## Delta

- Average reward: +0.011
- Success rate: +0.000
- Invalid move rate: +0.000

## Textual Gradients

- GuessTheNumber SLM episodes wasted turns or repeated guesses: Add a rule: For Guess The Number, track the current lower and upper bounds from every higher/lower hint, never repeat a previous guess, and choose the midpoint of the remaining interval.
