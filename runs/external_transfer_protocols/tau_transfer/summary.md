# tau_transfer Transfer Probe

Accepted TextGrad update: true
Gradient count: 5
Learned rule ids: check_policy, confirm_final_state, verify_identity

| Method | Source | Target | Episodes | Success | Reward | Invalid | Repeated | Turns | Updates |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_policy | tau-retail | tau-airline+tau-banking | 6 | 0.000 | 0.325 | 0.833 | 0.000 | 10.67 | 0 |
| textgrad_rl | tau-retail | tau-airline+tau-banking | 6 | 0.833 | 0.967 | 0.000 | 0.000 | 6.33 | 1 |
