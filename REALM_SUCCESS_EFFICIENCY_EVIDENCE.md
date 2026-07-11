# REALM Multi-Seed Evidence

Use [REALM_MULTISEED_CONFIDENCE_INTERVALS.md](/Users/jaray/Documents/TextGrad-RL/REALM_MULTISEED_CONFIDENCE_INTERVALS.md) as the current paper-facing result. TextWorldExpress is intentionally excluded.

Model: `qwen2.5:7b`; temperature: `0.7`; 10 paired seeds per task; 95% hierarchical paired bootstrap intervals with 10,000 resamples.

| Benchmark | Tasks x seeds | No TextGrad success | TextGrad-RL success | Improvement | Mean step decrease | Median step decrease |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TextArena | 12 x 10 | 16.7% [1.7, 36.7] | 14.2% [0.8, 33.3] | -2.5 pp [-10.0, +4.2] | +0.50 [-1.00, +9.00] | +0.00 [-1.00, +9.00] |
| TextWorld representative slice | 6 x 10 | 51.7% [16.7, 83.3] | 61.7% [25.0, 96.7] | +10.0 pp [-13.3, +35.0] | +0.14 [-6.44, +7.00] | +0.00 [-8.00, +8.00] |
| Task-weighted aggregate | 18 x 10 | 28.3% [11.1, 47.2] | 30.0% [11.7, 50.6] | +1.7 pp [-7.8, +11.7] | +0.25 [-4.51, +5.47] | +0.00 [-8.00, +8.00] |

The current result is not statistically conclusive: the pooled success-rate interval includes zero. TextGrad-RL improves the TextWorld slice descriptively, but the TextArena estimate is slightly negative and all step-efficiency intervals are wide.

TextWorld uses six preregistered representative tasks and a 20-action cap for this stochastic run: two simple objective tasks, one coin task, one treasure task, and two cooking tasks. It should be described as a multi-seed slice, not as a replacement for the full 24-task TextWorld evaluation.

Machine-readable outputs are in `runs/realm_multiseed_qwen7b_t07_10seed_slice/`, including paired episode records and `paired_bootstrap_summary.csv`.
