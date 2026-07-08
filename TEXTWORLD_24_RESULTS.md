# TextWorld 24-Game Local Benchmark Results

Microsoft TextWorld was run locally with `textworld==1.7.0`. The suite generates 24 `.z8` games from built-in challenge generators: 6 `tw-simple`, 6 `tw-coin_collector`, 6 `tw-treasure_hunter`, and 6 `tw-cooking` games. It uses admissible commands and visible observations/objectives, without oracle walkthrough commands.

Command:

```bash
bash scripts/run_textworld_24.sh
```

## Overall Results

| Method | Games | Reward | Success | Invalid | Repeated | Turns | Updates | Gradients |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_prompt | 24 | 0.484 | 0.292 | 0.000 | 0.750 | 47.00 | 0 | 0 |
| textgrad_policy_iteration | 24 | 0.583 | 0.542 | 0.000 | 0.792 | 44.21 | 1 | 3 |
| textgrad_ppo | 24 | 0.484 | 0.292 | 0.000 | 0.750 | 47.00 | 0 | 3 |

## Family Results

| Method | Family | Games | Reward | Success | Invalid | Repeated | Turns |
|---|---|---:|---:|---:|---:|---:|---:|
| fixed_prompt | tw-coin_collector | 6 | 1.000 | 1.000 | 0.000 | 0.667 | 11.50 |
| fixed_prompt | tw-cooking | 6 | 0.301 | 0.000 | 0.000 | 0.833 | 68.33 |
| fixed_prompt | tw-simple | 6 | 0.467 | 0.000 | 0.000 | 1.000 | 80.00 |
| fixed_prompt | tw-treasure_hunter | 6 | 0.167 | 0.167 | 0.000 | 0.500 | 28.17 |
| textgrad_policy_iteration | tw-coin_collector | 6 | 1.000 | 1.000 | 0.000 | 0.667 | 18.17 |
| textgrad_policy_iteration | tw-cooking | 6 | 0.167 | 0.167 | 0.000 | 1.000 | 72.17 |
| textgrad_policy_iteration | tw-simple | 6 | 0.167 | 0.000 | 0.000 | 1.000 | 80.00 |
| textgrad_policy_iteration | tw-treasure_hunter | 6 | 1.000 | 1.000 | 0.000 | 0.500 | 6.50 |
| textgrad_ppo | tw-coin_collector | 6 | 1.000 | 1.000 | 0.000 | 0.667 | 11.50 |
| textgrad_ppo | tw-cooking | 6 | 0.301 | 0.000 | 0.000 | 0.833 | 68.33 |
| textgrad_ppo | tw-simple | 6 | 0.467 | 0.000 | 0.000 | 1.000 | 80.00 |
| textgrad_ppo | tw-treasure_hunter | 6 | 0.167 | 0.167 | 0.000 | 0.500 | 28.17 |

## Gate Decisions

- `textgrad_policy_iteration` accepted one prompt-policy update. Validation score improved from `0.334` to `0.496`; learned rules covered objective-sequence parsing, graph exploration, and cooking recipes.
- `textgrad_ppo` generated the same candidate but rejected it because the behavioral trust-region proxy was slightly above threshold: `approx_kl=0.679` with `ppo_target_kl=0.65`. It therefore evaluated as the fixed prompt.

## Takeaway

TextWorld is feasible locally and adds a stronger generated text-adventure benchmark than TextWorldExpress. The current TextGrad-RL update transfers especially well to `tw-treasure_hunter` objective-navigation games, while `tw-simple` and `tw-cooking` expose remaining weaknesses in precise objective parsing and long-horizon recipe execution.
