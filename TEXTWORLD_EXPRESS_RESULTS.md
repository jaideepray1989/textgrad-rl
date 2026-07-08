# TextWorldExpress Local Benchmark Results

TextWorldExpress was run locally with the `textworld-express==1.1.0` package. This benchmark requires no API keys, no browser credentials, and no hosted services.

Command:

```bash
.venv/bin/python -m textgrad_rl.benchmarks.textworld_express_suite \
  --train-seeds 3 \
  --val-seeds 3 \
  --test-seeds 3 \
  --max-steps 80 \
  --output-dir runs/textworld_express_suite
```

## Overall Results

| Method | Games | Episodes | Reward | Success | Invalid | Repeated | Turns | Updates | Gradients |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_prompt | 8 | 24 | -0.261 | 0.167 | 0.000 | 0.667 | 37.12 | 0 | 0 |
| textgrad_policy_iteration | 8 | 24 | 0.792 | 0.708 | 0.000 | 0.583 | 28.17 | 1 | 8 |
| textgrad_ppo | 8 | 24 | -0.261 | 0.167 | 0.000 | 0.667 | 37.12 | 0 | 8 |

## Per-Game Success

| Game | fixed_prompt | textgrad_policy_iteration | textgrad_ppo |
|---|---:|---:|---:|
| arithmetic | 0.000 | 1.000 | 0.000 |
| coin | 0.333 | 0.333 | 0.333 |
| cookingworld | 0.000 | 0.333 | 0.000 |
| mapreader | 1.000 | 1.000 | 1.000 |
| peckingorder | 0.000 | 1.000 | 0.000 |
| simonsays | 0.000 | 1.000 | 0.000 |
| sorting | 0.000 | 1.000 | 0.000 |
| twc | 0.000 | 0.000 | 0.000 |

## Gate Decisions

- `textgrad_policy_iteration` accepted one update. Validation score improved from `-0.442` to `0.892`; 8 game-family rules were added.
- `textgrad_ppo` generated the same candidate update but rejected it because the behavioral trust-region proxy was high: `approx_kl=0.969` with `ppo_target_kl=0.65`. It therefore evaluated as the fixed prompt.

## Takeaway

TextWorldExpress is a useful no-credential local benchmark for this project. It adds instruction following, arithmetic/quantity reasoning, recipe planning, map navigation, and commonsense placement beyond TextArena. The current TextGrad-RL policy update helps strongly on instruction-following and quantity-reasoning families, partially helps cooking, and does not yet solve `twc` or improve `coin` exploration.
