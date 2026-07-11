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

RulePI crosses the interaction break-even point within the 240-game TextWorld study (about 202 deployments), but not within the 130-episode TextArena study (about 756 deployments). The efficiency claim is therefore deployment-time amortization, not uniformly lower total experimental compute.

## Ablation and Boundary

- Gated and ungated persistence have identical TextWorld success (56.7%) and test actions (42.25/game).
- On TextArena they also tie in success (61.5%); RulePI has slightly higher reward (0.573 vs 0.564). These runs do not establish a validation-gate advantage.
- With qwen2.5:7b at temperature 0.7, the no-TextGrad and RulePI protocols both score 25.0% success. The gate rejects the edit because validation score moves from 0.508 to 0.506, so the deployed RulePI policy remains the base policy.

The supported positive claim is therefore about persistent textual rules for prompt-aware structured actors, not generic improvement of small language-model agents and not a demonstrated benefit from validation gating.
