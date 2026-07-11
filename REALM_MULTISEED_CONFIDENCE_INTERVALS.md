# REALM Multi-Seed Confidence Intervals

Model: `qwen2.5:7b`; temperature: `0.7`. Intervals are 95% paired hierarchical bootstrap intervals (10,000 resamples), clustering by task and resampling seeds within task. TextWorldExpress is excluded.

| Benchmark | Tasks x seeds | No TextGrad success | TextGrad-RL success | Paired improvement (pp) | Common successes | Mean step decrease | Median step decrease |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TextArena | 12 x 10 | 16.7% [1.7, 36.7] | 14.2% [0.8, 33.3] | -2.5 pp [-10.0, +4.2] | 12 | +0.50 [-1.00, +9.00] | +0.00 [-1.00, +9.00] |
| TextWorld (6-task slice) | 6 x 10 | 51.7% [16.7, 83.3] | 61.7% [25.0, 96.7] | +10.0 pp [-13.3, +35.0] | 28 | +0.14 [-6.44, +7.00] | +0.00 [-8.00, +8.00] |
| Task-weighted aggregate | 18 x 10 | 28.3% [11.1, 47.2] | 30.0% [11.7, 50.6] | +1.7 pp [-7.8, +11.7] | 40 | +0.25 [-4.51, +5.47] | +0.00 [-8.00, +8.00] |

Positive step decrease means TextGrad-RL used fewer actions. Step statistics include only paired episodes solved by both methods.
Macro-average success-rate improvement across the two benchmark rows: +3.8 percentage points.

## Protocol

- TextArena evaluates the previously train/validation-selected prompt policies on new held-out environment seeds.
- TextWorld (6-task slice) evaluates the base and structured TextGrad controller rules with Qwen ranking controller-proposed actions and a 20-action cap.
- Pairing uses identical task and environment seed across methods. Ollama does not expose a generation RNG seed through this runner, so paired rollouts do not use common model randomness.
