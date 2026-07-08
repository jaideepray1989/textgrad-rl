# Compiled Results by Benchmark, Policy, and Model

This is the paper-facing aggregate table across the completed local benchmark suite. Rows are held-out test aggregates unless the source report explicitly marks the run as a transfer-protocol probe. PPO rows are intentionally excluded from this compiled artifact because the latest runs show that the PPO-style gate usually rejects the useful prompt edit or underperforms the simpler gated update.

Method names are normalized across benchmark harnesses:

- `fixed_actor`, `fixed_policy`, and `fixed_prompt_slm` are reported as `fixed_prompt`.
- `textgrad_rl` and `textgrad_policy_iteration_slm` are reported as `textgrad_policy_iteration`.
- The 30-seed SLM candidate-pool run keeps `textgrad_rl_no_gate` and `textgrad_rl_train_val` separate because those are distinct non-PPO ablations.

The matching machine-readable table is `COMPILED_RESULTS_BY_BENCHMARK_POLICY_MODEL.csv`.

## Main Local Benchmarks

| Benchmark | Suite | Model | Temp. | Policy | Tasks | Episodes | Reward | Score | Success | Invalid | Repeated | Trunc. | Turns | Updates |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TextArena Broad 50 | all | prompt-aware heuristic actor | n/a | fixed_prompt | 50 | 195 | 0.094 | n/a | 0.185 | 0.518 | 0.749 | n/a | 9.32 | 0 |
| TextArena Broad 50 | all | prompt-aware heuristic actor | n/a | textgrad_policy_iteration | 50 | 195 | 0.261 | n/a | 0.369 | 0.451 | 0.718 | n/a | 8.50 | 6 |
| TextArena Difficulty Generalization | all | prompt-aware heuristic actor | n/a | fixed_prompt | n/a | n/a | 0.190 | n/a | 0.167 | 0.139 | n/a | n/a | 22.597 | 0 |
| TextArena Difficulty Generalization | all | prompt-aware heuristic actor | n/a | textgrad_policy_iteration | n/a | n/a | 0.897 | n/a | 0.833 | 0.000 | n/a | n/a | 16.958 | n/a |
| BrowserGym MiniWoB++ 50 | all | prompt-aware heuristic browser actor | n/a | fixed_prompt | 50 | 150 | 0.200 | n/a | 0.200 | 0.020 | 0.713 | n/a | 4.05 | 0 |
| BrowserGym MiniWoB++ 50 | all | prompt-aware heuristic browser actor | n/a | textgrad_policy_iteration | 50 | 150 | 0.273 | n/a | 0.273 | 0.020 | 0.640 | n/a | 3.86 | 1 |
| BrowserGym MiniWoB++ 10 | all | gpt-oss:20b | 0.7 | fixed_prompt | 10 | 30 | 0.933 | n/a | 0.933 | 0.000 | 0.333 | n/a | 2.73 | 0 |
| BrowserGym MiniWoB++ 10 | all | gpt-oss:20b | 0.7 | textgrad_policy_iteration | 10 | 30 | 0.933 | n/a | 0.933 | 0.000 | 0.300 | n/a | 2.47 | 0 |
| TextWorldExpress 8 | all | prompt-aware heuristic text-game actor | n/a | fixed_prompt | 8 | 24 | -0.261 | n/a | 0.167 | 0.000 | 0.667 | n/a | 37.12 | 0 |
| TextWorldExpress 8 | all | prompt-aware heuristic text-game actor | n/a | textgrad_policy_iteration | 8 | 24 | 0.792 | n/a | 0.708 | 0.000 | 0.583 | n/a | 28.17 | 1 |
| TextWorld 24 | all | prompt-aware heuristic text-game actor | n/a | fixed_prompt | 24 | 24 | 0.484 | n/a | 0.292 | 0.000 | 0.750 | n/a | 47.00 | 0 |
| TextWorld 24 | all | prompt-aware heuristic text-game actor | n/a | textgrad_policy_iteration | 24 | 24 | 0.583 | n/a | 0.542 | 0.000 | 0.792 | n/a | 44.21 | 1 |

## SLM Stress Tests

| Benchmark | Suite | Model | Temp. | Policy | Tasks | Episodes | Reward | Score | Success | Invalid | Repeated | Trunc. | Turns | Updates |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TextArena SLM Candidate Pool 30-Seed | puzzle | qwen2.5:3b | 0.7 | fixed_prompt | 4 | 120 | 0.068 | -0.447 | 0.000 | 1.000 | n/a | 0.000 | 3.058 | 0 |
| TextArena SLM Candidate Pool 30-Seed | puzzle | qwen2.5:3b | 0.7 | textgrad_rl_no_gate | 4 | 120 | 0.075 | -0.426 | 0.008 | 0.975 | n/a | 0.008 | 3.200 | 1 |
| TextArena SLM Candidate Pool 30-Seed | puzzle | qwen2.5:3b | 0.7 | textgrad_rl_train_val | 4 | 120 | 0.069 | -0.440 | 0.000 | 0.983 | n/a | 0.008 | 3.200 | 0 |
| TextArena SLM Candidate Pool 30-Seed | social | qwen2.5:3b | 0.7 | fixed_prompt | 3 | 90 | -0.311 | -0.345 | 0.256 | 0.144 | n/a | 0.000 | 17.878 | 0 |
| TextArena SLM Candidate Pool 30-Seed | social | qwen2.5:3b | 0.7 | textgrad_rl_no_gate | 3 | 90 | -0.289 | -0.311 | 0.278 | 0.144 | n/a | 0.000 | 17.778 | 1 |
| TextArena SLM Candidate Pool 30-Seed | social | qwen2.5:3b | 0.7 | textgrad_rl_train_val | 3 | 90 | -0.122 | -0.089 | 0.378 | 0.133 | n/a | 0.000 | 17.867 | 1 |
| TextArena SLM Candidate Pool 30-Seed | real_slm | qwen2.5:3b | 0.7 | fixed_prompt | 5 | 150 | 0.323 | 0.214 | 0.227 | 0.300 | n/a | 0.167 | 6.207 | 0 |
| TextArena SLM Candidate Pool 30-Seed | real_slm | qwen2.5:3b | 0.7 | textgrad_rl_no_gate | 5 | 150 | 0.328 | 0.245 | 0.247 | 0.273 | n/a | 0.160 | 5.913 | 1 |
| TextArena SLM Candidate Pool 30-Seed | real_slm | qwen2.5:3b | 0.7 | textgrad_rl_train_val | 5 | 150 | 0.350 | 0.237 | 0.233 | 0.307 | n/a | 0.180 | 6.207 | 1 |
| TextArena SLM Single-Seed | puzzle | gpt-oss:20b | default | fixed_prompt | 4 | n/a | 0.025 | -0.488 | 0.000 | 1.000 | n/a | 0.000 | 2.500 | n/a |
| TextArena SLM Single-Seed | puzzle | gpt-oss:20b | default | textgrad_policy_iteration | 4 | n/a | 0.050 | -0.464 | 0.000 | 1.000 | n/a | 0.000 | 2.750 | n/a |
| TextArena SLM Single-Seed | social | gpt-oss:20b | default | fixed_prompt | 3 | n/a | 0.333 | 0.413 | 0.333 | 0.000 | n/a | 0.000 | 17.333 | n/a |
| TextArena SLM Single-Seed | social | gpt-oss:20b | default | textgrad_policy_iteration | 3 | n/a | 0.333 | 0.413 | 0.333 | 0.000 | n/a | 0.000 | 17.333 | n/a |
| TextArena SLM Single-Seed | real_slm | gpt-oss:20b | default | fixed_prompt | 5 | n/a | -0.084 | -0.598 | 0.000 | 1.000 | n/a | 0.000 | 2.800 | n/a |
| TextArena SLM Single-Seed | real_slm | gpt-oss:20b | default | textgrad_policy_iteration | 5 | n/a | -0.084 | -0.598 | 0.000 | 1.000 | n/a | 0.000 | 2.800 | n/a |

## Local Transfer Protocol Probes

These rows are not official WebArena, tau-bench, or SWE-bench leaderboard results. They are local source-to-target transfer probes using deterministic task contracts.

| Benchmark | Suite | Model | Temp. | Policy | Tasks | Episodes | Reward | Success | Invalid | Repeated | Turns | Updates |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Local Transfer Protocol Probe | browser_transfer | deterministic protocol simulator | n/a | fixed_prompt | 6 | 6 | 0.250 | 0.000 | 1.000 | 0.167 | 10.17 | 0 |
| Local Transfer Protocol Probe | browser_transfer | deterministic protocol simulator | n/a | textgrad_policy_iteration | 6 | 6 | 0.944 | 0.833 | 0.000 | 0.000 | 6.83 | 1 |
| Local Transfer Protocol Probe | tau_transfer | deterministic protocol simulator | n/a | fixed_prompt | 6 | 6 | 0.325 | 0.000 | 0.833 | 0.000 | 10.67 | 0 |
| Local Transfer Protocol Probe | tau_transfer | deterministic protocol simulator | n/a | textgrad_policy_iteration | 6 | 6 | 0.967 | 0.833 | 0.000 | 0.000 | 6.33 | 1 |
| Local Transfer Protocol Probe | swe_transfer | deterministic protocol simulator | n/a | fixed_prompt | 5 | 5 | 0.273 | 0.000 | 0.400 | 0.000 | 9.40 | 0 |
| Local Transfer Protocol Probe | swe_transfer | deterministic protocol simulator | n/a | textgrad_policy_iteration | 5 | 5 | 0.960 | 0.800 | 0.000 | 0.000 | 5.40 | 1 |

## Readout

TextGrad-RL is strongest where the actor can execute the learned textual rule: TextArena Broad 50 improves success from 0.185 to 0.369, TextWorldExpress from 0.167 to 0.708, TextWorld 24 from 0.292 to 0.542, and MiniWoB++ 50 from 0.200 to 0.273. The 30-seed qwen2.5:3b SLM run shows the harder frontier: action formatting dominates puzzle failures, while train/validation gating helps social tasks and improves real-SLM reward but does not uniformly improve success.

Blocked official WebArena preflight artifacts are not included as scored benchmark rows because no task scores were produced.

