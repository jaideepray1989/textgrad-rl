# BrowserGym MiniWoB++ Results

This run evaluates BrowserGym MiniWoB++ as a lower-infrastructure browser benchmark than WebArena. Unlike WebArena, it only requires local static MiniWoB++ HTML files plus Playwright Chromium.

## Setup Used

- Python runtime: `.venv_miniwob` with Python 3.12
- Package: `browsergym-miniwob==0.14.3`
- MiniWoB++ HTML root:

```bash
MINIWOB_URL=file:///tmp/miniwob-plusplus/miniwob/html/miniwob/
```

10-task smoke command:

```bash
scripts/run_miniwob_subset.sh runs/miniwob_subset_10x3
```

50-task suite command:

```bash
MINIWOB_ENVS=50 scripts/run_miniwob_subset.sh runs/miniwob_subset_50x3
```

## 10-Task Smoke Suite

10 MiniWoB++ tasks:

| Task |
|---|
| click-button |
| enter-text |
| focus-text |
| click-checkboxes |
| click-option |
| choose-list |
| click-test |
| click-dialog |
| click-tab |
| login-user |

Each method was evaluated on 3 test seeds per task, for 30 test episodes per method.

## Results

### 10 Tasks x 3 Seeds

| Method | Episodes | Success | Invalid Action | Repeated Action | Avg Turns | Accepted Prompt Updates |
|---|---:|---:|---:|---:|---:|---:|
| fixed_actor | 30 | 0.600 | 0.000 | 0.300 | 2.40 | 0 |
| textgrad_rl | 30 | 0.900 | 0.000 | 0.000 | 1.60 | 1 |
| textgrad_rl_ppo | 30 | 0.600 | 0.000 | 0.300 | 2.40 | 0 |

### 50 Tasks x 3 Seeds

The larger suite covers 50 MiniWoB++ tasks: 8 clicking tasks, 9 selection tasks, 10 text-entry tasks, 6 form tasks, 6 menu/navigation tasks, 8 simulated-app tasks, and 3 reading tasks. Coordinate-heavy drag/draw tasks are intentionally excluded because the current actor uses accessibility-tree actions rather than mouse-coordinate control.

| Method | Episodes | Success | Invalid Action | Repeated Action | Avg Turns | Accepted Prompt Updates |
|---|---:|---:|---:|---:|---:|---:|
| fixed_actor | 150 | 0.200 | 0.020 | 0.713 | 4.05 | 0 |
| textgrad_rl | 150 | 0.273 | 0.020 | 0.640 | 3.86 | 1 |
| textgrad_rl_ppo | 150 | 0.200 | 0.020 | 0.713 | 4.05 | 0 |

50-task category breakdown:

| Method | Category | Envs | Episodes | Success | Invalid Action | Repeated Action | Avg Turns |
|---|---|---:|---:|---:|---:|---:|---:|
| fixed_actor | clicking | 8 | 24 | 0.500 | 0.000 | 0.500 | 3.00 |
| fixed_actor | forms | 6 | 18 | 0.222 | 0.167 | 0.611 | 3.89 |
| fixed_actor | menu_navigation | 6 | 18 | 0.167 | 0.000 | 0.833 | 4.33 |
| fixed_actor | reading | 3 | 9 | 0.000 | 0.000 | 0.667 | 3.67 |
| fixed_actor | selection | 9 | 27 | 0.000 | 0.000 | 0.889 | 5.00 |
| fixed_actor | simulated_app | 8 | 24 | 0.000 | 0.000 | 1.000 | 5.00 |
| fixed_actor | text_entry | 10 | 30 | 0.367 | 0.000 | 0.500 | 3.30 |
| textgrad_rl | clicking | 8 | 24 | 0.500 | 0.000 | 0.500 | 3.00 |
| textgrad_rl | forms | 6 | 18 | 0.222 | 0.167 | 0.611 | 3.89 |
| textgrad_rl | menu_navigation | 6 | 18 | 0.167 | 0.000 | 0.833 | 4.33 |
| textgrad_rl | reading | 3 | 9 | 0.000 | 0.000 | 0.667 | 3.67 |
| textgrad_rl | selection | 9 | 27 | 0.407 | 0.000 | 0.481 | 3.96 |
| textgrad_rl | simulated_app | 8 | 24 | 0.000 | 0.000 | 1.000 | 5.00 |
| textgrad_rl | text_entry | 10 | 30 | 0.367 | 0.000 | 0.500 | 3.30 |
| textgrad_rl_ppo | clicking | 8 | 24 | 0.500 | 0.000 | 0.500 | 3.00 |
| textgrad_rl_ppo | forms | 6 | 18 | 0.222 | 0.167 | 0.611 | 3.89 |
| textgrad_rl_ppo | menu_navigation | 6 | 18 | 0.167 | 0.000 | 0.833 | 4.33 |
| textgrad_rl_ppo | reading | 3 | 9 | 0.000 | 0.000 | 0.667 | 3.67 |
| textgrad_rl_ppo | selection | 9 | 27 | 0.000 | 0.000 | 0.889 | 5.00 |
| textgrad_rl_ppo | simulated_app | 8 | 24 | 0.000 | 0.000 | 1.000 | 5.00 |
| textgrad_rl_ppo | text_entry | 10 | 30 | 0.367 | 0.000 | 0.500 | 3.30 |

## Interpretation

The fixed actor solves direct click and text-entry tasks but fails selection tasks such as `click-checkboxes`, `click-option`, and `choose-list`: it selects the right option and then repeats that action instead of clicking Submit.

`textgrad_rl` observes this repeated-action failure during training, proposes the rule:

> After selecting all required checkbox, radio, or list options, click Submit exactly once instead of repeating the selected option.

The validation gate accepts this rule, improving test success from `0.600` to `0.900`, eliminating repeated actions, and reducing average turns from `2.40` to `1.60`.

On the 50-task suite, the same rule improves overall success from `0.200` to `0.273`. The improvement is concentrated in selection tasks: success rises from `0.000` to `0.407`, repeated-action rate falls from `0.889` to `0.481`, and average turns fall from `5.00` to `3.96`.

`textgrad_rl_ppo` proposes the same candidate rule, but rejects it because the short base prompt makes the appended rule look large under the current character-level KL proxy. In the 10-task run:

- `old_score`: `-0.20`
- `new_score`: `1.48`
- `advantage`: `1.68`
- `clipped_surrogate`: `2.016`
- `kl_proxy`: `0.819`

In the 50-task run, PPO rejected the candidate on a validation tie:

- `old_score`: `-0.20`
- `new_score`: `-0.20`
- `advantage`: `0.00`
- `clipped_surrogate`: `0.00`
- `kl_proxy`: `0.819`

This is a useful diagnostic: MiniWoB++ confirms TextGrad-RL can improve browser-control behavior with much less infrastructure than WebArena, but also shows that the current PPO-style KL proxy can be too conservative for short prompts and too dependent on a small validation slice.

## Artifacts

- `runs/miniwob_subset_10x3/episodes.jsonl`
- `runs/miniwob_subset_10x3/prompt_updates.jsonl`
- `runs/miniwob_subset_10x3/summary.csv`
- `runs/miniwob_subset_10x3/summary.json`
- `runs/miniwob_subset_10x3/summary.md`
- `runs/miniwob_subset_50x3/episodes.jsonl`
- `runs/miniwob_subset_50x3/prompt_updates.jsonl`
- `runs/miniwob_subset_50x3/summary.csv`
- `runs/miniwob_subset_50x3/summary.json`
- `runs/miniwob_subset_50x3/summary.md`
- `runs/miniwob_subset_50x3/category_summary.csv`
- `runs/miniwob_subset_50x3/category_summary.json`
- `runs/miniwob_subset_50x3/category_summary.md`
