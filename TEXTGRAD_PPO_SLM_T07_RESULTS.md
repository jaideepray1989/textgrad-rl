# TextGrad-PPO-SLM at Temperature 0.7

This run adds `textgrad_ppo_slm`, a PPO-style trust-region gate for frozen-SLM TextArena prompt updates.

Unlike deterministic TextGrad-PPO, this SLM variant matters because the actor is stochastic at nonzero temperature. Candidate prompt edits can improve one split while degrading another, so PPO-SLM uses paired train/validation rollouts, a clipped behavioral-ratio surrogate, and invalid/truncation guards before accepting a text update.

## Command

```bash
.venv/bin/python -m textgrad_rl.benchmarks.textarena_expanded_suites \
  --suites puzzle,social,real_slm \
  --slm-methods fixed_prompt_slm,textgrad_policy_iteration_slm,textgrad_ppo_slm \
  --slm-train-seeds 1 \
  --slm-val-seeds 1 \
  --slm-test-seeds 1 \
  --output-dir runs/textarena_ppo_slm_qwen25_3b_t07 \
  --model qwen2.5:3b \
  --temperature 0.7 \
  --timeout 90
```

## Results

### Puzzle SLM

| method | reward | score | success | invalid | truncated | turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | 0.146 | -0.372 | 0.000 | 1.000 | 0.000 | 3.750 |
| textgrad_policy_iteration_slm | 0.121 | -0.396 | 0.000 | 1.000 | 0.000 | 3.500 |
| textgrad_ppo_slm | 0.146 | -0.375 | 0.000 | 1.000 | 0.000 | 4.250 |

### Social SLM

| method | reward | score | success | invalid | truncated | turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | -0.333 | -0.417 | 0.333 | 0.333 | 0.000 | 16.667 |
| textgrad_policy_iteration_slm | 0.000 | -0.083 | 0.333 | 0.333 | 0.000 | 16.667 |
| textgrad_ppo_slm | -0.333 | -0.417 | 0.333 | 0.333 | 0.000 | 16.667 |

### Real SLM

| method | reward | score | success | invalid | truncated | turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | 0.180 | 0.106 | 0.200 | 0.200 | 0.200 | 4.800 |
| textgrad_policy_iteration_slm | -0.020 | -0.255 | 0.000 | 0.200 | 0.400 | 7.000 |
| textgrad_ppo_slm | 0.180 | 0.104 | 0.200 | 0.200 | 0.200 | 5.200 |

## Interpretation

Temperature matters. At `temperature=0.7`, the same prompt policy can produce different actions across train, validation, and test. That makes one-seed SLM acceptance noisy.

In this full run, `textgrad_ppo_slm` rejected all candidate edits. That is useful behavior: it avoided the real-SLM regression seen in `textgrad_policy_iteration_slm`, where GuessTheNumber degraded from success to truncation. On social, however, the stricter PPO gate also rejected a candidate that policy iteration kept and that happened to improve IPD reward on test.

So the current story is:

- `textgrad_ppo_slm` is a conservative guardrail under stochastic LLM behavior.
- `textgrad_policy_iteration_slm` can get lucky wins at `t=0.7`, but it can also accept unstable edits.
- With only one train/validation/test seed, stochastic acceptance is underpowered.

For a stronger paper run, use at least 3-5 SLM seeds per split at `temperature=0.7`, or keep `temperature=0.0` for deterministic prompt-policy evaluation and use `temperature=0.7` as a robustness/stress suite.

## Artifacts

- `runs/textarena_ppo_slm_qwen25_3b_t07/puzzle_slm/summary.md`
- `runs/textarena_ppo_slm_qwen25_3b_t07/social_slm/summary.md`
- `runs/textarena_ppo_slm_qwen25_3b_t07/real_slm/summary.md`
- `runs/textarena_ppo_slm_qwen25_3b_t07/*_slm/update_decisions.jsonl`
