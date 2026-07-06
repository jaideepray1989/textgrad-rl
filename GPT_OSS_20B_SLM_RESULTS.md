# GPT-OSS 20B TextArena SLM Results

Reproduction command:

```bash
.venv/bin/python -m textgrad_rl.benchmarks.textarena_expanded_suites \
  --suites puzzle,social,real_slm \
  --slm-train-seeds 1 \
  --slm-val-seeds 1 \
  --slm-test-seeds 1 \
  --output-dir runs/textarena_expanded_suites_gpt_oss_20b \
  --model gpt-oss:20b \
  --timeout 180
```

Runtime was about 49 minutes on the local Ollama backend.

## Puzzle SLM

| method | reward | score | success | invalid | truncated | turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | 0.025 | -0.488 | 0.000 | 1.000 | 0.000 | 2.500 |
| scalar_prompt_search_slm | 0.025 | -0.486 | 0.000 | 1.000 | 0.000 | 2.250 |
| modular_textgrad_slm | 0.025 | -0.488 | 0.000 | 1.000 | 0.000 | 2.500 |
| textgrad_rl_plus_slm | 0.025 | -0.488 | 0.000 | 1.000 | 0.000 | 2.500 |
| textgrad_policy_iteration_slm | 0.050 | -0.464 | 0.000 | 1.000 | 0.000 | 2.750 |

## Social SLM

| method | reward | score | success | invalid | truncated | turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | 0.333 | 0.413 | 0.333 | 0.000 | 0.000 | 17.333 |
| scalar_prompt_search_slm | 0.333 | 0.413 | 0.333 | 0.000 | 0.000 | 17.333 |
| modular_textgrad_slm | 0.333 | 0.418 | 0.333 | 0.000 | 0.000 | 16.333 |
| textgrad_rl_plus_slm | 0.333 | 0.413 | 0.333 | 0.000 | 0.000 | 17.333 |
| textgrad_policy_iteration_slm | 0.333 | 0.413 | 0.333 | 0.000 | 0.000 | 17.333 |

## Real SLM

| method | reward | score | success | invalid | truncated | turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | -0.084 | -0.598 | 0.000 | 1.000 | 0.000 | 2.800 |
| scalar_prompt_search_slm | -0.084 | -0.597 | 0.000 | 1.000 | 0.000 | 2.600 |
| modular_textgrad_slm | -0.084 | -0.598 | 0.000 | 1.000 | 0.000 | 2.800 |
| textgrad_rl_plus_slm | -0.084 | -0.598 | 0.000 | 1.000 | 0.000 | 2.800 |
| textgrad_policy_iteration_slm | -0.084 | -0.598 | 0.000 | 1.000 | 0.000 | 2.800 |

## Output Health

| suite | episodes | invalid episodes | raw outputs | empty outputs | empty rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| puzzle_slm | 84 | 84 | 220 | 176 | 0.800 |
| social_slm | 63 | 4 | 1054 | 620 | 0.588 |
| real_slm | 105 | 105 | 268 | 207 | 0.772 |

## Notes

`gpt-oss:20b` improved the social suite versus the earlier `qwen2.5:3b` run by avoiding invalid moves and winning Kuhn Poker, but it regressed puzzle and real-SLM. The dominant failure mode is not model size; it is action formatting. On the actual TextArena observations, `gpt-oss:20b` often returned an empty final answer through Ollama's OpenAI-compatible endpoint, which the runner normalized to `[0]`. A direct sanity probe returned `[Cooperate]` correctly, so the client is reading the right response field.

For a paper-quality `gpt-oss:20b` run, the next fix should be a stricter action-format adapter: retry empty outputs, prefer bracketed legal actions from the observation, and add environment-specific output schemas before scoring the policy updates.

Artifacts:

- `runs/textarena_expanded_suites_gpt_oss_20b/puzzle_slm/summary.md`
- `runs/textarena_expanded_suites_gpt_oss_20b/social_slm/summary.md`
- `runs/textarena_expanded_suites_gpt_oss_20b/real_slm/summary.md`
