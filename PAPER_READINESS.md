# Paper Readiness Report

This repository now contains runnable artifacts for the seven paper-readiness items.

## 1. Real Frozen SLM Experiments

Implemented `textgrad_rl.benchmarks.textarena_slm_compare`, which runs a local OpenAI-compatible frozen SLM on TextArena `GuessTheNumber-v0`. The only optimized object is a text variable; model weights stay fixed.

Reproduced run:

```bash
.venv/bin/python -m textgrad_rl.benchmarks.textarena_slm_compare \
  --model qwen2.5:3b \
  --train-seeds 3 \
  --val-seeds 3 \
  --test-seeds 5 \
  --seed 9300 \
  --output-dir runs/textarena_slm_qwen25_3b_guess_number
```

Held-out result: no TextGrad reward `0.905`, TextGrad-RL reward `0.916`; both had `0.800` success and `0.000` invalid-move rate. The validation gate accepted one prompt update.

## 2. Main Baselines

The TextArena paper suite compares:

- `fixed_prompt`
- `scalar_prompt_search`
- `monolithic_textgrad`
- `modular_textgrad`
- `no_acceptance_gate`

The real-SLM benchmark compares the same frozen local model under `no_textgrad` and `textgrad_rl`.

## 3. Train/Validation/Test Discipline

`textarena_paper_suite` and `textarena_slm_compare` both separate train, validation, and held-out test seeds. TextGrad candidates are evaluated on validation seeds before test evaluation.

## 4. Stronger Benchmark Suite

The TextArena suite covers ten offline environments:

`Nim-v0`, `ConnectFour-v0`, `ReverseTicTacToe-v0`, `GuessTheNumber-v0`, `FrozenLake-v0`, `TowerOfHanoi-v0`, `LightsOut-v0`, `Mastermind-v0`, `Blackjack-v0`, and `Bandit-v0`.

Reproduced run:

```bash
.venv/bin/python -m textgrad_rl.benchmarks.textarena_paper_suite \
  --repetitions 3 \
  --train-seeds 5 \
  --val-seeds 5 \
  --test-seeds 10 \
  --output-dir runs/textarena_paper_suite
```

## 5. Ablations Proving Mechanism

The full suite includes scalar-only, monolithic, modular, and no-gate ablations. In the current deterministic TextArena suite, modular, monolithic, and no-gate tie; that should be reported as a limitation. The scalar-only baseline stays at fixed-prompt performance, supporting the value of textual trajectory feedback over scalar outcome mutation.

## 6. Statistical Rigor

`runs/textarena_paper_suite/bootstrap_cis.csv` reports bootstrap 95% confidence intervals over 390 held-out test episodes per method.

Headline held-out means:

| method | reward | success | invalid |
| --- | ---: | ---: | ---: |
| fixed_prompt | 0.185 | 0.205 | 0.105 |
| scalar_prompt_search | 0.185 | 0.205 | 0.105 |
| modular_textgrad | 0.563 | 0.615 | 0.000 |

Reward bootstrap CIs:

| method | mean | 95% CI |
| --- | ---: | ---: |
| fixed_prompt | 0.185 | [0.118, 0.245] |
| scalar_prompt_search | 0.185 | [0.124, 0.245] |
| modular_textgrad | 0.563 | [0.503, 0.623] |

## 7. Qualitative Evidence

Qualitative artifacts are written to:

- `runs/textarena_paper_suite/qualitative_examples.md`
- `runs/textarena_paper_suite/methods/*/rep_*/gradients.json`
- `runs/textarena_slm_qwen25_3b_guess_number/games/*.jsonl`

These include failure modes, trajectory evidence, suggested textual edits, accepted update metadata, and raw model outputs for the frozen-SLM benchmark.

## Remaining Publication Gaps

The prototype is much stronger now, but a compelling paper should still add broader real-SLM coverage: more local SLM models, more TextArena environments with real SLM actors, larger held-out seed counts, and a noisier critic setting where the acceptance gate separates from no-gate more clearly.

## Follow-Up: TextGrad-RL+

`textgrad_rl.benchmarks.textarena_textgrad_plus` adds multi-candidate edits, causal assignment, replay validation, bootstrap-gated acceptance, and a learned rule library.

The 10-environment TextArena run in `runs/textarena_textgrad_plus` improves modular TextGrad-RL from `0.563` to `0.583` held-out reward over 390 episodes per method, while preserving `0.615` success and `0.000` invalid moves. The result table is in `TEXTGRAD_PLUS_RESULTS.md`.

## Follow-Up: TextGrad Policy Iteration

`textgrad_rl.benchmarks.textarena_policy_iteration` strengthens the RL framing with action-level credit assignment, advantage-weighted textual gradients, candidate policy search, replay buffers, and a tabular value critic.

The 10-environment run in `runs/textarena_policy_iteration` ties TextGrad-RL+ on held-out reward (`0.583`) but makes the policy-improvement mechanism explicit. All accepted rules are selected as `advantage_weighted_env_rule` candidates. The result table is in `TEXTGRAD_POLICY_ITERATION_RESULTS.md`.

## Follow-Up: Expanded TextArena Benchmarks

`textgrad_rl.benchmarks.textarena_expanded_suites` adds difficulty-generalization, puzzle-SLM, social-SLM, and real-SLM benchmark suites.

The run in `runs/textarena_expanded_suites` shows strong rule transfer on hard TextArena variants (`0.897` reward, `0.833` success, `0.000` invalid moves) and exposes harder local-SLM gaps on puzzle/social tasks. The result table is in `EXPANDED_TEXTARENA_RESULTS.md`.
