# TextGrad PPO SLM Temperature 0.7, 5-Seed Results

This run repeats the SLM benchmark at temperature 0.7 with five environment seeds per
train/validation/test split, so each suite has more samples for accepting or rejecting
TextGrad prompt edits.

Command:

```bash
.venv/bin/python -m textgrad_rl.benchmarks.textarena_expanded_suites \
  --suites puzzle,social,real_slm \
  --slm-methods fixed_prompt_slm,textgrad_policy_iteration_slm,textgrad_ppo_slm \
  --slm-train-seeds 5 \
  --slm-val-seeds 5 \
  --slm-test-seeds 5 \
  --output-dir runs/textarena_ppo_slm_qwen25_3b_t07_5seed \
  --model qwen2.5:3b \
  --temperature 0.7 \
  --timeout 90
```

Artifacts:

- `runs/textarena_ppo_slm_qwen25_3b_t07_5seed/puzzle_slm/summary.md`
- `runs/textarena_ppo_slm_qwen25_3b_t07_5seed/social_slm/summary.md`
- `runs/textarena_ppo_slm_qwen25_3b_t07_5seed/real_slm/summary.md`
- `runs/textarena_ppo_slm_qwen25_3b_t07_5seed/*_slm/update_decisions.jsonl`

## Test Results

| suite | method | episodes | reward | score | success | invalid | truncated | turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| puzzle | fixed_prompt_slm | 20 | 0.107 | -0.409 | 0.000 | 1.000 | 0.000 | 3.150 |
| puzzle | textgrad_policy_iteration_slm | 20 | 0.062 | -0.454 | 0.000 | 1.000 | 0.000 | 3.300 |
| puzzle | textgrad_ppo_slm | 20 | 0.054 | -0.461 | 0.000 | 1.000 | 0.000 | 3.000 |
| social | fixed_prompt_slm | 15 | -0.467 | -0.586 | 0.200 | 0.267 | 0.000 | 17.200 |
| social | textgrad_policy_iteration_slm | 15 | -0.733 | -0.985 | 0.067 | 0.400 | 0.000 | 17.000 |
| social | textgrad_ppo_slm | 15 | -0.600 | -0.785 | 0.067 | 0.267 | 0.000 | 17.067 |
| real_slm | fixed_prompt_slm | 25 | 0.380 | 0.245 | 0.200 | 0.320 | 0.160 | 7.120 |
| real_slm | textgrad_policy_iteration_slm | 25 | 0.377 | 0.296 | 0.240 | 0.240 | 0.200 | 6.240 |
| real_slm | textgrad_ppo_slm | 25 | 0.493 | 0.533 | 0.360 | 0.120 | 0.200 | 6.160 |

Macro averages over the three suites:

| method | reward | score | success | invalid |
| --- | ---: | ---: | ---: | ---: |
| fixed_prompt_slm | 0.007 | -0.250 | 0.133 | 0.529 |
| textgrad_policy_iteration_slm | -0.098 | -0.381 | 0.102 | 0.547 |
| textgrad_ppo_slm | -0.018 | -0.238 | 0.142 | 0.462 |

## Update Decisions

| suite | method | accepted | val score delta | train score delta | PPO surrogate delta | PPO KL | note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| puzzle | textgrad_policy_iteration_slm | yes | +0.013 | +0.021 | n/a | n/a | Accepted by score/invalid/truncation gate, but test score regressed vs fixed prompt. |
| puzzle | textgrad_ppo_slm | no | +0.030 | +0.009 | -0.000 | 0.001 | Rejected because clipped surrogate delta was negative despite validation score improvement. |
| social | textgrad_policy_iteration_slm | no | +0.031 | -0.132 | n/a | n/a | Rejected because the train split regressed. |
| social | textgrad_ppo_slm | no | -0.161 | -0.266 | -0.095 | 0.027 | Rejected on validation, train, surrogate, and invalid-rate regression. |
| real_slm | textgrad_policy_iteration_slm | no | +0.256 | -0.055 | n/a | n/a | Rejected because train score and invalid rate regressed. |
| real_slm | textgrad_ppo_slm | no | -0.099 | +0.156 | -0.031 | 0.022 | Rejected because validation and surrogate regressed. |

## Interpretation

The stronger 5-seed run makes the main story sharper: `textgrad_ppo_slm` is acting as a
conservative acceptance gate, not as a source of accepted prompt improvements yet. PPO
accepted zero edits across puzzle, social, and real-SLM. Plain policy iteration accepted
one puzzle edit, but that edit did not generalize to the held-out test rollouts.

The real-SLM test table shows `textgrad_ppo_slm` with the best test score and success
rate, but that should not be presented as causal PPO improvement. PPO rejected its real-SLM
candidate edit, so the better test result comes from stochastic generation at temperature
0.7 under the same base prompt family rather than from an accepted prompt update. The 5
environment seeds reduce task-seed noise, but they do not fully remove model-sampling noise.

The useful paper-facing result is therefore methodological: PPO-style clipping and
surrogate checks prevented accepting brittle TextGrad edits that looked good on parts of
the data. To make a stronger empirical claim, the next run should repeat each final test
method with multiple independent model-sampling replications, or force a fixed decoding
seed if the local model server supports it.
