# SLM Candidate-Pool Extension Results

This update implements the first main-conference extension path: a wider candidate generator plus a 30-seed stochastic evaluation recipe.

## Implemented

- Multi-candidate SLM prompt-policy updates via `--slm-candidate-count`.
- Optional stronger candidate generator model via `--slm-candidate-model`.
- Per-candidate logging in `candidate_decisions.jsonl`.
- Selected update logging in `update_decisions.jsonl`.
- Optional chat-completion logprob capture via `--slm-request-logprobs`.
- Open-weight exact action-ratio scaffolding in `textgrad_rl.benchmarks.action_probability`.
- External trajectory adapter for WebArena/SWE-bench style benchmarks.

## Smoke Run

Command:

```bash
.venv/bin/python -m textgrad_rl.benchmarks.textarena_expanded_suites \
  --suites real_slm \
  --slm-methods textgrad_rl_ppo_slm \
  --slm-train-seeds 1 \
  --slm-val-seeds 1 \
  --slm-test-seeds 1 \
  --slm-candidate-count 2 \
  --output-dir runs/textarena_slm_candidate_pool_smoke \
  --model qwen2.5:3b \
  --temperature 0.7 \
  --timeout 90
```

Result:

| method | reward | score | success | invalid | truncated | turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| textgrad_rl_ppo_slm | 0.380 | 0.343 | 0.200 | 0.000 | 0.400 | 7.400 |

Candidate decisions:

| candidate | source | accepted | selected | val score delta | surrogate delta | invalid delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| candidate_000_textgrad | textgrad_default | false | true | +0.377 | +0.000 | +0.000 |
| candidate_001_general_textarena_slm_policy | heuristic_rule | false | false | +0.002 | -0.012 | +0.100 |

The smoke verifies that the runner evaluates multiple candidates, logs all candidate decisions, and keeps the old policy when PPO rejects every candidate. This is expected: the first candidate improved validation score but missed the minimum clipped-surrogate threshold; the second increased invalid moves and had negative surrogate delta.

## Recommended 30-Seed Run

```bash
bash scripts/run_slm_candidate_pool_30seed.sh
```

Default script settings:

- actor model: `qwen2.5:3b`
- candidate generator: `gpt-oss:20b`
- suites: puzzle, social, real_slm
- methods: fixed prompt, no-gate, train/val gate, full PPO
- candidate count: 8
- train/validation/test seeds: 30 each
- temperature: 0.7

This long run was not executed in this update because it is expected to be substantially slower than the previous 5-seed SLM experiments.
