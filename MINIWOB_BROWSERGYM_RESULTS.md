# BrowserGym MiniWoB++ Results

This run evaluates BrowserGym MiniWoB++ as a lower-infrastructure browser benchmark than WebArena. Unlike WebArena, it only requires local static MiniWoB++ HTML files plus Playwright Chromium.

## Setup Used

- Python runtime: `.venv_miniwob` with Python 3.12
- Package: `browsergym-miniwob==0.14.3`
- MiniWoB++ HTML root:

```bash
MINIWOB_URL=file:///tmp/miniwob-plusplus/miniwob/html/miniwob/
```

Run command:

```bash
scripts/run_miniwob_subset.sh runs/miniwob_subset_10x3
```

## Task Suite

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

| Method | Episodes | Success | Invalid Action | Repeated Action | Avg Turns | Accepted Prompt Updates |
|---|---:|---:|---:|---:|---:|---:|
| fixed_actor | 30 | 0.600 | 0.000 | 0.300 | 2.40 | 0 |
| textgrad_rl | 30 | 0.900 | 0.000 | 0.000 | 1.60 | 1 |
| textgrad_rl_ppo | 30 | 0.600 | 0.000 | 0.300 | 2.40 | 0 |

## Interpretation

The fixed actor solves direct click and text-entry tasks but fails selection tasks such as `click-checkboxes`, `click-option`, and `choose-list`: it selects the right option and then repeats that action instead of clicking Submit.

`textgrad_rl` observes this repeated-action failure during training, proposes the rule:

> After selecting all required checkbox, radio, or list options, click Submit exactly once instead of repeating the selected option.

The validation gate accepts this rule, improving test success from `0.600` to `0.900`, eliminating repeated actions, and reducing average turns from `2.40` to `1.60`.

`textgrad_rl_ppo` proposes the same candidate rule, but rejects it because the short base prompt makes the appended rule look large under the current character-level KL proxy:

- `old_score`: `-0.20`
- `new_score`: `1.48`
- `advantage`: `1.68`
- `clipped_surrogate`: `2.016`
- `kl_proxy`: `0.819`

This is a useful diagnostic: MiniWoB++ confirms TextGrad-RL can improve browser-control behavior with much less infrastructure than WebArena, but also shows that the current PPO-style KL proxy can be too conservative for short prompts.

## Artifacts

- `runs/miniwob_subset_10x3/episodes.jsonl`
- `runs/miniwob_subset_10x3/prompt_updates.jsonl`
- `runs/miniwob_subset_10x3/summary.csv`
- `runs/miniwob_subset_10x3/summary.json`
- `runs/miniwob_subset_10x3/summary.md`
