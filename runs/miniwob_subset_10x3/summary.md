# BrowserGym MiniWoB++ Results

Tasks: 10 (click-button, enter-text, focus-text, click-checkboxes, click-option, choose-list, click-test, click-dialog, click-tab, login-user)
Test seeds: 3
Max steps: 5

| Method | Episodes | Success | Invalid action | Repeated action | Avg turns | Accepted updates |
|---|---:|---:|---:|---:|---:|---:|
| fixed_actor | 30 | 0.600 | 0.000 | 0.300 | 2.40 | 0 |
| textgrad_rl | 30 | 0.900 | 0.000 | 0.000 | 1.60 | 1 |
| textgrad_rl_ppo | 30 | 0.600 | 0.000 | 0.300 | 2.40 | 0 |
