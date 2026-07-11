# TextWorld 24 qwen2.5:7b Full-Budget Rerun

Model: `qwen2.5:7b` through the local Ollama OpenAI-compatible endpoint.
Temperature: `0.7`.

This rerun removes the previous shallow 3-step probe and uses the standard local TextWorld budget of 80 steps per game. To keep one slow local model response from blocking the full benchmark, each action call uses a hard `curl --max-time 5` wall-clock timeout. Timed-out calls are recorded as parse/invalid failures.

Command:

```bash
.venv/bin/python -m textgrad_rl.benchmarks.textworld_slm_suites \
  --suites textworld_24 \
  --model qwen2.5:7b \
  --temperature 0.7 \
  --max-tokens 16 \
  --max-steps 80 \
  --timeout 120 \
  --model-call-timeout 5 \
  --min-mean-delta 0.001 \
  --textworld-train-per-family 1 \
  --textworld-val-per-family 1 \
  --output-dir runs/qwen25_7b_textworld24_full80_t5
```

## Overall Results

| method | problems | reward | score | success | invalid | parse fail | repeated | turns | update accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_textgrad | 24 | 0.303 | 0.097 | 0.250 | 0.042 | 0.042 | 0.833 | 55.67 | 0 |
| textgrad_rl | 24 | 0.296 | 0.131 | 0.250 | 0.000 | 0.000 | 0.750 | 53.88 | 0 |

## TextGrad-RL Delta

| metric | delta |
| --- | ---: |
| reward | -0.007 |
| score | +0.034 |
| success | +0.000 |
| invalid | -0.042 |
| parse fail | -0.042 |
| repeated | -0.083 |
| turns | -1.79 |

TextGrad-RL improves the shaped score slightly by reducing invalid/timeout failures and repeated-action episodes, but it does not improve success.

## Family Results

| family | method | reward | success | invalid | repeated | turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| tw-simple | no_textgrad | 0.089 | 0.000 | 0.167 | 1.000 | 72.83 |
| tw-simple | textgrad_rl | 0.033 | 0.000 | 0.000 | 1.000 | 80.00 |
| tw-coin_collector | no_textgrad | 0.500 | 0.500 | 0.000 | 0.833 | 41.83 |
| tw-coin_collector | textgrad_rl | 0.500 | 0.500 | 0.000 | 0.667 | 41.50 |
| tw-treasure_hunter | no_textgrad | 0.500 | 0.500 | 0.000 | 0.500 | 41.00 |
| tw-treasure_hunter | textgrad_rl | 0.500 | 0.500 | 0.000 | 0.500 | 41.00 |
| tw-cooking | no_textgrad | 0.124 | 0.000 | 0.000 | 1.000 | 67.00 |
| tw-cooking | textgrad_rl | 0.152 | 0.000 | 0.000 | 0.833 | 53.00 |

## Gate Decision

The TextGrad-RL update was rejected:

| old validation score | new validation score | gradients | accepted |
| ---: | ---: | ---: | --- |
| 0.508 | 0.506 | 3 | false |

The candidate rule reduced some repetition but did not improve validation score, so the train/validation gate correctly kept the original prompt.

## What Prevents qwen2.5:7b + TextGrad-RL From Working Here?

1. The model usually emits legal actions, but not goal-directed action sequences. Format is mostly solved; planning is not.
2. The hard failures are valid no-progress loops. Example from `simple_1_dense_detailed`: `examine antique trunk` 31 times, `take old key from antique trunk` 19 times, and `insert old key into antique trunk` 18 times. The needed action was to use the key on the wooden door, then continue the objective sequence.
3. TextWorld 24 requires persistent symbolic state: inventory, object containment, door/key relations, map position, and the next unmet clause in a long natural-language objective. The current prompt-only qwen actor does not maintain that state reliably.
4. TextGrad-RL proposes broad textual rules, but the actor does not reliably operationalize them into the next admissible command. The validation gate rejects the update because the candidate does not improve held-out validation.
5. The train/validation signal is sparse in this SLM run: one training and one validation game per family. That is enough to diagnose failure modes but not enough to learn robust family-specific policies.
6. Longer uncapped runs expose latency as an engineering problem. Some local qwen calls take long enough to require a hard per-action timeout.

## Takeaway

Removing the shallow step cap did not make TextGrad-RL succeed on TextWorld 24. It revealed the real bottleneck: qwen2.5:7b needs an explicit stateful planner or memory/controller layer over admissible actions. TextGrad-RL can improve textual policy rules, but the actor must be capable of executing those rules over long horizons.

Artifacts:

- Summary: `runs/qwen25_7b_textworld24_full80_t5/textworld_24/summary.md`
- Per-problem records: `runs/qwen25_7b_textworld24_full80_t5/textworld_24/no_textgrad/test.jsonl`
- TextGrad records: `runs/qwen25_7b_textworld24_full80_t5/textworld_24/textgrad_rl/test.jsonl`
