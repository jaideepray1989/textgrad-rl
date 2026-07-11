# RulePI Paper Evidence

## Claim 1: Persistent textual rules improve controlled interactive agents

| Benchmark | Fixed | Diagnostic retry | Ungated persistence | RulePI | RulePI - fixed (95% CI) |
|---|---:|---:|---:|---:|---:|
| TextWorld-24, 10 generations | 33.8% | 55.4% | 56.7% | 56.7% | +22.9 pp [+17.1, +29.2] |
| TextArena supported-10, 10 seeds | 19.2% | 61.5% | 61.5% | 61.5% | +42.3 pp [+18.0, +72.0] |

## Claim 2: Persistence amortizes task-local diagnosis

| Benchmark | Retry attempts | RulePI attempts | Retry test actions | RulePI test actions | RulePI optimization | Break-even tasks |
|---|---:|---:|---:|---:|---:|---:|
| TextWorld-24 | 1.66 | 1.00 | 81.78 | 42.25 | 7965 | 202 |
| TextArena supported-10 | 1.81 | 1.00 | 20.68 | 10.19 | 7921 | 756 |

RulePI crosses the interaction break-even point within the 240-instance TextWorld study (about 202 deployments), but not within the 130-instance TextArena study (about 756 deployments). The efficiency claim is therefore deployment-time amortization, not uniformly lower total experimental compute.

## Ablation and Boundary

- Gated and ungated persistence have identical TextWorld success (56.7%) and test actions (42.25/instance).
- On TextArena they also tie in success (61.5%); RulePI has slightly higher reward (0.573 vs 0.564). These runs do not establish a validation-gate advantage.

The supported positive claim is therefore about persistent textual rules for prompt-aware structured actors, not generic improvement of language-model agents and not a demonstrated benefit from validation gating.

## Reviewer Controls

| Benchmark | Method | Success | Attempts | Test actions |
|---|---|---:|---:|---:|
| TextWorld-24 | RulePI | 56.7% | 1.00 | 42.25 |
| TextWorld-24 | Always-on rules | 56.7% | 1.00 | 42.25 |
| TextWorld-24 | Retry with persistence | 56.7% | 1.48 | 78.94 |
| TextArena supported-10 | RulePI, 6 selected rules | 61.5% | 1.00 | 10.19 |
| TextArena supported-10 | Always-on, 12 rules | 61.5% | 1.00 | 10.78 |
| TextArena supported-10 | Random 6 environment rules | 42.8% | 1.00 | 11.19 |
| TextArena supported-10 | Retry with persistence | 61.5% | 1.43 | 17.28 |

The TextArena random control averages 30 independently sampled six-rule policies on the same 130 test instances. Policy-level success ranges from 30.8% to 53.8% (SD 6.7 percentage points). TextWorld always-on is identical to RulePI because all three available rules are selected in every repetition; TextWorld therefore does not isolate a trajectory-selection benefit.

## Gate Sensitivity and Iteration Boundary

Using saved TextWorld validation trajectories, the default gate accepts 10/10 updates at delta 0.001, 9/10 at delta 0.05, and 6/10 at delta 0.20. Corresponding counterfactual test success is 56.7%, 54.2%, and 46.2%. A per-family protected-regression check accepts 7/10 and yields 49.2% test success. These are post-hoc comparisons on the existing eight-instance validation sets, not fresh confirmatory runs.

A deterministic second pass over the fixed TextWorld library adds no rule: rule count remains 3 and the text-variable version remains 1. Larger or freely generated libraries were not tested.

Machine-readable outputs and the full random-policy selections are under `runs/rulepi_reviewer_controls/`; the reproducible runner is `scripts/run_rulepi_reviewer_controls.py`.
