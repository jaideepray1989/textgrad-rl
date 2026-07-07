# BrowserGym MiniWoB++ Results

Tasks: 50 (click-button, click-button-sequence, click-checkboxes, click-checkboxes-large, click-checkboxes-soft, click-checkboxes-transfer, click-dialog, click-dialog-2, click-link, click-menu, click-menu-2, click-option, click-scroll-list, click-tab, click-tab-2, click-test, click-test-2, click-widget, choose-list, copy-paste, copy-paste-2, email-inbox, email-inbox-delete, email-inbox-forward, email-inbox-important, email-inbox-noscroll, email-inbox-reply, enter-date, enter-password, enter-text, enter-text-2, enter-text-dynamic, enter-time, find-word, focus-text, focus-text-2, form-sequence, form-sequence-2, form-sequence-3, login-user, login-user-popup, navigate-tree, number-checkboxes, read-table, read-table-2, search-engine, sign-agreement, social-media, use-autocomplete, use-autocomplete-nodelay)
Test seeds: 3
Max steps: 5

| Method | Episodes | Success | Invalid action | Repeated action | Avg turns | Accepted updates |
|---|---:|---:|---:|---:|---:|---:|
| fixed_actor | 150 | 0.200 | 0.020 | 0.713 | 4.05 | 0 |
| textgrad_rl | 150 | 0.273 | 0.020 | 0.640 | 3.86 | 1 |
| textgrad_rl_ppo | 150 | 0.200 | 0.020 | 0.713 | 4.05 | 0 |
