# Experimental Results

This file presents the checked-in results grouped per benchmark. Rows are held-out test aggregates unless a section explicitly says it is a local transfer-protocol probe. PPO rows are omitted from this paper-facing summary because the current PPO-style gate generally rejected useful prompt edits or underperformed the simpler gated update.

Policy names are normalized across harnesses:

- `fixed_actor`, `fixed_policy`, and `fixed_prompt_slm` are reported as `fixed_prompt`.
- `textgrad_rl` and `textgrad_policy_iteration_slm` are reported as `textgrad_policy_iteration`.
- The 30-seed SLM candidate-pool run keeps `textgrad_rl_no_gate` and `textgrad_rl_train_val` separate because they are distinct non-PPO ablations.

## TextArena Broad 50

| Policy | Model | Tasks | Episodes | Reward | Success | Invalid | Repeated | Turns | Updates |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_prompt | prompt-aware heuristic actor | 50 | 195 | 0.094 | 0.185 | 0.518 | 0.749 | 9.32 | 0 |
| textgrad_policy_iteration | prompt-aware heuristic actor | 50 | 195 | 0.261 | 0.369 | 0.451 | 0.718 | 8.50 | 6 |

Source: `BROAD_50_BENCHMARK_RESULTS.md`.

## TextArena Difficulty Generalization

| Policy | Model | Reward | Success | Invalid | Turns | Updates |
|---|---|---:|---:|---:|---:|---:|
| fixed_prompt | prompt-aware heuristic actor | 0.190 | 0.167 | 0.139 | 22.597 | 0 |
| textgrad_policy_iteration | prompt-aware heuristic actor | 0.897 | 0.833 | 0.000 | 16.958 | n/a |

Source: `EXPANDED_TEXTARENA_RESULTS.md`.

## BrowserGym MiniWoB++ 50

| Policy | Model | Tasks | Episodes | Reward | Success | Invalid | Repeated | Turns | Updates |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_prompt | prompt-aware heuristic browser actor | 50 | 150 | 0.200 | 0.200 | 0.020 | 0.713 | 4.05 | 0 |
| textgrad_policy_iteration | prompt-aware heuristic browser actor | 50 | 150 | 0.273 | 0.273 | 0.020 | 0.640 | 3.86 | 1 |

Source: `MINIWOB_BROWSERGYM_RESULTS.md`.

## BrowserGym MiniWoB++ 10

| Policy | Model | Temp. | Tasks | Episodes | Reward | Success | Invalid | Repeated | Turns | Updates |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_prompt | gpt-oss:20b | 0.7 | 10 | 30 | 0.933 | 0.933 | 0.000 | 0.333 | 2.73 | 0 |
| textgrad_policy_iteration | gpt-oss:20b | 0.7 | 10 | 30 | 0.933 | 0.933 | 0.000 | 0.300 | 2.47 | 0 |

Source: `MINIWOB_BROWSERGYM_RESULTS.md`.

## TextWorldExpress 8

| Policy | Model | Games | Episodes | Reward | Success | Invalid | Repeated | Turns | Updates |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_prompt | prompt-aware heuristic text-game actor | 8 | 24 | -0.261 | 0.167 | 0.000 | 0.667 | 37.12 | 0 |
| textgrad_policy_iteration | prompt-aware heuristic text-game actor | 8 | 24 | 0.792 | 0.708 | 0.000 | 0.583 | 28.17 | 1 |

Source: `TEXTWORLD_EXPRESS_RESULTS.md`.

## TextWorld 24

| Policy | Model | Test games | Success [95% CI] | Reward | Attempts/task | Test actions/task |
|---|---|---:|---:|---:|---:|---:|
| fixed_prompt | prompt-aware heuristic text-game actor | 240 | 33.8% [27.5, 40.0] | 0.528 | 1.00 | 44.15 |
| retry_with_diagnostics | prompt-aware heuristic text-game actor | 240 | 55.4% [48.8, 62.5] | 0.604 | 1.66 | 81.78 |
| ungated_persistent_rules | prompt-aware heuristic text-game actor | 240 | 56.7% [50.0, 62.9] | 0.617 | 1.00 | 42.25 |
| textgrad_policy_iteration (RulePI) | prompt-aware heuristic text-game actor | 240 | 56.7% [50.0, 63.3] | 0.617 | 1.00 | 42.25 |

Across 10 disjoint generations, RulePI improves success over fixed by 22.9 percentage points (paired 95% CI `[17.1, 29.2]`) and uses 39.5 fewer test actions per game than diagnostic retry (`[-45.0, -34.2]`). Gated and ungated persistence are tied, so the positive result is attributable to persistent rules rather than validation gating.

Source: `TEXTWORLD_24_RESULTS.md`.

## TextArena Supported 10

| Policy | Model | Episodes | Success | Reward | Attempts/task | Test actions/task | Optimization actions |
|---|---|---:|---:|---:|---:|---:|---:|
| fixed_prompt | prompt-aware heuristic actor | 130 | 0.192 | 0.167 | 1.00 | 12.14 | 0 |
| retry_with_diagnostics | prompt-aware heuristic actor | 130 | 0.615 | 0.550 | 1.81 | 20.68 | 0 |
| ungated_persistent_rules | prompt-aware heuristic actor | 130 | 0.615 | 0.564 | 1.00 | 10.19 | 7,921 |
| textgrad_policy_iteration (RulePI) | prompt-aware heuristic actor | 130 | 0.615 | 0.573 | 1.00 | 10.19 | 7,921 |

RulePI improves paired success over fixed by 42.3 percentage points (hierarchical paired-bootstrap CI `[18.0, 72.0]`). All adaptive methods tie in success; persistent policies avoid the second test attempt required by diagnostic retry. RulePI and ungated persistence also tie in success, with only a small reward difference. At the observed action rates, RulePI needs about 756 deployments to amortize its 7,921 optimization actions; the 130-episode run therefore demonstrates deployment-time, not total-compute, efficiency.

Source: `RULEPI_PAPER_EVIDENCE.md`.

## TextArena SLM Candidate Pool 30-Seed

| Suite | Policy | Model | Temp. | Tasks | Episodes | Reward | Score | Success | Invalid | Trunc. | Turns | Updates |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| puzzle | fixed_prompt | qwen2.5:3b | 0.7 | 4 | 120 | 0.068 | -0.447 | 0.000 | 1.000 | 0.000 | 3.058 | 0 |
| puzzle | textgrad_rl_no_gate | qwen2.5:3b | 0.7 | 4 | 120 | 0.075 | -0.426 | 0.008 | 0.975 | 0.008 | 3.200 | 1 |
| puzzle | textgrad_rl_train_val | qwen2.5:3b | 0.7 | 4 | 120 | 0.069 | -0.440 | 0.000 | 0.983 | 0.008 | 3.200 | 0 |
| social | fixed_prompt | qwen2.5:3b | 0.7 | 3 | 90 | -0.311 | -0.345 | 0.256 | 0.144 | 0.000 | 17.878 | 0 |
| social | textgrad_rl_no_gate | qwen2.5:3b | 0.7 | 3 | 90 | -0.289 | -0.311 | 0.278 | 0.144 | 0.000 | 17.778 | 1 |
| social | textgrad_rl_train_val | qwen2.5:3b | 0.7 | 3 | 90 | -0.122 | -0.089 | 0.378 | 0.133 | 0.000 | 17.867 | 1 |
| real_slm | fixed_prompt | qwen2.5:3b | 0.7 | 5 | 150 | 0.323 | 0.214 | 0.227 | 0.300 | 0.167 | 6.207 | 0 |
| real_slm | textgrad_rl_no_gate | qwen2.5:3b | 0.7 | 5 | 150 | 0.328 | 0.245 | 0.247 | 0.273 | 0.160 | 5.913 | 1 |
| real_slm | textgrad_rl_train_val | qwen2.5:3b | 0.7 | 5 | 150 | 0.350 | 0.237 | 0.233 | 0.307 | 0.180 | 6.207 | 1 |

Sources: `runs/textarena_slm_candidate_pool_30seed/*/summary.md`.

## TextArena SLM Single-Seed

| Suite | Policy | Model | Reward | Score | Success | Invalid | Trunc. | Turns |
|---|---|---|---:|---:|---:|---:|---:|---:|
| puzzle | fixed_prompt | gpt-oss:20b | 0.025 | -0.488 | 0.000 | 1.000 | 0.000 | 2.500 |
| puzzle | textgrad_policy_iteration | gpt-oss:20b | 0.050 | -0.464 | 0.000 | 1.000 | 0.000 | 2.750 |
| social | fixed_prompt | gpt-oss:20b | 0.333 | 0.413 | 0.333 | 0.000 | 0.000 | 17.333 |
| social | textgrad_policy_iteration | gpt-oss:20b | 0.333 | 0.413 | 0.333 | 0.000 | 0.000 | 17.333 |
| real_slm | fixed_prompt | gpt-oss:20b | -0.084 | -0.598 | 0.000 | 1.000 | 0.000 | 2.800 |
| real_slm | textgrad_policy_iteration | gpt-oss:20b | -0.084 | -0.598 | 0.000 | 1.000 | 0.000 | 2.800 |

Source: `GPT_OSS_20B_SLM_RESULTS.md`.

## Local Transfer Protocol Probe

These are local source-to-target transfer probes, not official WebArena, tau-bench, or SWE-bench leaderboard scores.

| Suite | Policy | Model | Tasks | Episodes | Reward | Success | Invalid | Repeated | Turns | Updates |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| browser_transfer | fixed_prompt | deterministic protocol simulator | 6 | 6 | 0.250 | 0.000 | 1.000 | 0.167 | 10.17 | 0 |
| browser_transfer | textgrad_policy_iteration | deterministic protocol simulator | 6 | 6 | 0.944 | 0.833 | 0.000 | 0.000 | 6.83 | 1 |
| tau_transfer | fixed_prompt | deterministic protocol simulator | 6 | 6 | 0.325 | 0.000 | 0.833 | 0.000 | 10.67 | 0 |
| tau_transfer | textgrad_policy_iteration | deterministic protocol simulator | 6 | 6 | 0.967 | 0.833 | 0.000 | 0.000 | 6.33 | 1 |
| swe_transfer | fixed_prompt | deterministic protocol simulator | 5 | 5 | 0.273 | 0.000 | 0.400 | 0.000 | 9.40 | 0 |
| swe_transfer | textgrad_policy_iteration | deterministic protocol simulator | 5 | 5 | 0.960 | 0.800 | 0.000 | 0.000 | 5.40 | 1 |

Source: `EXTERNAL_TRANSFER_PROTOCOL_RESULTS.md`.

## Summary Readout

The strongest controlled evidence now comes from the matched baselines. Across 10 TextWorld generations, persistent rules improve success from 33.8% to 56.7% with a paired interval fully above zero. On the supported TextArena suite, success improves from 19.2% to 61.5%. Persistent policies match task-local diagnostic retry while using roughly half as many test actions. Validation gating does not improve success over ungated persistence in either benchmark. The qwen2.5:7b boundary run remains at 25.0% success and rejects its candidate update, so the positive claim is limited to prompt-aware structured actors.
