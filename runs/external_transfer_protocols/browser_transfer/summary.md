# browser_transfer Transfer Probe

Accepted TextGrad update: true
Gradient count: 3
Learned rule ids: avoid_repeated_browser_action, search_before_detail, select_then_submit

| Method | Source | Target | Episodes | Success | Reward | Invalid | Repeated | Turns | Updates |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_policy | MiniWoB | WebArena+WorkArena | 6 | 0.000 | 0.250 | 1.000 | 0.167 | 10.17 | 0 |
| textgrad_rl | MiniWoB | WebArena+WorkArena | 6 | 0.833 | 0.944 | 0.000 | 0.000 | 6.83 | 1 |
