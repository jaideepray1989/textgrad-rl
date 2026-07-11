# TextWorld 24-Game Local Benchmark Results

Microsoft TextWorld was run locally with `textworld==1.7.0`. Each of 10 repetitions generates 24 `.z8` test games: 6 `tw-simple`, 6 `tw-coin_collector`, 6 `tw-treasure_hunter`, and 6 `tw-cooking` games. Every repetition also uses 8 separately seeded training games and 8 separately seeded validation games; neither split overlaps its test games. All methods use admissible commands and visible observations/objectives without oracle walkthrough commands.

Command:

```bash
python scripts/run_rulepi_textworld_multiseed.py \
  --repetitions 10 \
  --max-steps 80 \
  --bootstrap-samples 10000 \
  --output-dir runs/rulepi_textworld_10seed
```

`retry_with_diagnostics` runs the fixed policy once and retries each failed test task once with task-local diagnostic rules. It does not retain rules across tasks or use validation. `ungated_persistent_rules` retains training-derived rules without validation. RulePI gates the same persistent rules on disjoint validation games and evaluates each test task once.

## Overall Results

| Method | Test games | Reward | Success [95% CI] | Attempts/task | Test actions/task | Optimization actions |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed policy | 240 | 0.528 | 33.8% [27.5, 40.0] | 1.00 | 44.15 | 0 |
| Retry + Diagnostics | 240 | 0.604 | 55.4% [48.8, 62.5] | 1.66 | 81.78 | 0 |
| Ungated persistent rules | 240 | 0.617 | 56.7% [50.0, 62.9] | 1.00 | 42.25 | 2,645 |
| RulePI | 240 | 0.617 | 56.7% [50.0, 63.3] | 1.00 | 42.25 | 7,965 |

Relative to fixed policy, RulePI improves paired success by **22.9 percentage points** (95% hierarchical paired-bootstrap CI `[17.1, 29.2]`). Relative to Retry + Diagnostics, it improves success by 1.25 points (`[0.0, 3.3]`) while reducing test interaction by 39.5 actions per game (`[-45.0, -34.2]`). Gated and ungated persistence are exactly tied in success, reward, and test actions, so this experiment supports persistence but not a validation-gate advantage.

## Example-Generation Family Results

The following family breakdown is for the original `seed=62001` generation and is retained as a qualitative example; aggregate claims use all 10 generations above.

| Method | Family | Games | Reward | Success | Test actions/task |
|---|---|---:|---:|---:|---:|
| fixed_prompt | tw-coin_collector | 6 | 1.000 | 1.000 | 11.50 |
| fixed_prompt | tw-cooking | 6 | 0.301 | 0.000 | 68.33 |
| fixed_prompt | tw-simple | 6 | 0.467 | 0.000 | 80.00 |
| fixed_prompt | tw-treasure_hunter | 6 | 0.167 | 0.167 | 28.17 |
| retry_with_diagnostics | tw-coin_collector | 6 | 1.000 | 1.000 | 11.50 |
| retry_with_diagnostics | tw-cooking | 6 | 0.167 | 0.167 | 140.50 |
| retry_with_diagnostics | tw-simple | 6 | 0.167 | 0.000 | 160.00 |
| retry_with_diagnostics | tw-treasure_hunter | 6 | 1.000 | 1.000 | 34.50 |
| textgrad_policy_iteration | tw-coin_collector | 6 | 1.000 | 1.000 | 18.17 |
| textgrad_policy_iteration | tw-cooking | 6 | 0.167 | 0.167 | 72.17 |
| textgrad_policy_iteration | tw-simple | 6 | 0.167 | 0.000 | 80.00 |
| textgrad_policy_iteration | tw-treasure_hunter | 6 | 1.000 | 1.000 | 6.50 |

## Gate Decisions

- RulePI accepted one policy update. On the disjoint validation games, score improved from `0.859` to `1.037`. The learned policy contains rules for objective-sequence parsing, graph exploration, and cooking recipes.
- Retry + Diagnostics diagnosed 17 failed first attempts and generated 31 task-local rules. No rule persisted to another task.
- Across 10 generations, RulePI accepted its candidate every time. Because ungated persistence produced the same test policy and outcomes, validation added evaluation cost without changing deployment behavior in this suite.

## Takeaway

Persistent textual rules substantially outperform the fixed controller and match or slightly exceed task-local diagnostic retry while roughly halving test-time interaction. The validation gate does not improve TextWorld outcomes over ungated persistence. The gains concentrate in `tw-treasure_hunter`; `tw-simple` and harder cooking games remain challenging.
