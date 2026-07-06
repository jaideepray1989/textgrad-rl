# Expanded TextArena Benchmark Results

This run covers the requested benchmark additions:

1. TextArena difficulty generalization.
2. TextArena puzzle suite.
3. TextArena multi-agent/social suite.
4. Real frozen-SLM TextArena suite.

Reproduction command:

```bash
.venv/bin/python -m textgrad_rl.benchmarks.textarena_expanded_suites \
  --suites difficulty,puzzle,social,real_slm \
  --repetitions 2 \
  --train-seeds 3 \
  --val-seeds 3 \
  --test-seeds 3 \
  --slm-train-seeds 1 \
  --slm-val-seeds 1 \
  --slm-test-seeds 1 \
  --output-dir runs/textarena_expanded_suites \
  --model qwen2.5:3b
```

## Difficulty Generalization

| method | reward | success | invalid | turns |
| --- | ---: | ---: | ---: | ---: |
| fixed_prompt | 0.190 | 0.167 | 0.139 | 22.597 |
| modular_textgrad | 0.830 | 0.792 | 0.000 | 18.236 |
| textgrad_rl_plus | 0.880 | 0.792 | 0.000 | 17.250 |
| textgrad_policy_iteration | 0.897 | 0.833 | 0.000 | 16.958 |

## Puzzle SLM

| method | reward | score | success | invalid | truncated | turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | 0.146 | -0.374 | 0.000 | 1.000 | 0.000 | 4.000 |
| scalar_prompt_search_slm | 0.146 | -0.374 | 0.000 | 1.000 | 0.000 | 4.000 |
| modular_textgrad_slm | 0.111 | -0.407 | 0.000 | 1.000 | 0.000 | 3.500 |
| textgrad_rl_plus_slm | 0.146 | -0.374 | 0.000 | 1.000 | 0.000 | 4.000 |
| textgrad_policy_iteration_slm | 0.111 | -0.407 | 0.000 | 1.000 | 0.000 | 3.500 |

## Social SLM

| method | reward | score | success | invalid | truncated | turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | -0.333 | -0.417 | 0.333 | 0.333 | 0.000 | 16.667 |
| scalar_prompt_search_slm | -0.333 | -0.417 | 0.333 | 0.333 | 0.000 | 16.667 |
| modular_textgrad_slm | -0.333 | -0.417 | 0.333 | 0.333 | 0.000 | 16.667 |
| textgrad_rl_plus_slm | -0.333 | -0.585 | 0.000 | 0.333 | 0.000 | 17.000 |
| textgrad_policy_iteration_slm | -0.333 | -0.585 | 0.000 | 0.333 | 0.000 | 17.000 |

## Real SLM

| method | reward | score | success | invalid | truncated | turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | 0.180 | 0.104 | 0.200 | 0.200 | 0.200 | 5.200 |
| scalar_prompt_search_slm | -0.020 | -0.254 | 0.000 | 0.200 | 0.400 | 6.800 |
| modular_textgrad_slm | 0.180 | 0.104 | 0.200 | 0.200 | 0.200 | 5.200 |
| textgrad_rl_plus_slm | 0.180 | 0.104 | 0.200 | 0.200 | 0.200 | 5.200 |
| textgrad_policy_iteration_slm | 0.180 | 0.104 | 0.200 | 0.200 | 0.200 | 5.200 |

## Artifact Links

- Difficulty generalization: `runs/textarena_expanded_suites/difficulty_generalization/summary.md`
- Puzzle SLM: `runs/textarena_expanded_suites/puzzle_slm/summary.md`
- Social SLM: `runs/textarena_expanded_suites/social_slm/summary.md`
- Real SLM: `runs/textarena_expanded_suites/real_slm/summary.md`

## Notes

The difficulty-generalization comparison is strong: `textgrad_policy_iteration` is best overall with `0.897` reward, `0.833` success, and no invalid moves.

The SLM suites are intentionally harsh. On these tiny qwen2.5:3b runs, TextGrad policy iteration does not beat the fixed-prompt SLM baseline; it trails puzzle slightly, regresses social on success/score, and ties real-SLM after the validation gate rejects harmful edits. These are useful next paper benchmarks because they expose where textual policy iteration needs stronger action-format grounding, better state tracking, more validation seeds, and model-scale comparisons.
