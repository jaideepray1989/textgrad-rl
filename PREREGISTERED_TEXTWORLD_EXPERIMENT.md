# Preregistered TextWorld TextGrad-RL Experiment

## Question

Can automatically generated, validation-gated textual policy updates improve a
frozen local language-model actor on held-out procedural TextWorld instances?

## Primary Protocol

- Actor: `qwen2.5:14b` at temperature `0.0`.
- Candidate generator: `gpt-oss:20b` at temperature `0.7`.
- Outer repetitions: `3`, each with independently generated train, validation,
  and test game seeds.
- Training: two task configurations per TextWorld family (eight episodes).
- Validation: one disjoint task configuration per family (four episodes).
- Test: all 24 task configurations, generated with a disjoint seed.
- Candidate pool: eight generator-produced, general policy-rule candidates.
- Environment horizon: 80 actions. A 300-second HTTP watchdog is retained only
  to surface infrastructure hangs; watchdog failures remain scored episodes.

## Methods

1. `fixed_prompt`: initial actor policy without an update.
2. `ungated_textgrad`: candidate zero from the automatically generated pool,
   selected before validation evaluation.
3. `validation_gated_textgrad`: the validation-best candidate only when its
   validation score is at least the fixed prompt's score and neither invalid nor
   parse-failure rate increases; otherwise the fixed prompt is retained.

Candidate generation receives only train trajectories. Candidate ranking and
acceptance receive only validation trajectories. Test trajectories are read only
after the final policy is frozen. Rules mentioning hidden state, seeds, oracle
walkthroughs, benchmark identifiers, or task-specific command strings are
rejected before validation.

## Primary Outcome And Decision Rule

The primary outcome is held-out task success. The primary comparison is
`validation_gated_textgrad - fixed_prompt`; the secondary comparison is
`validation_gated_textgrad - ungated_textgrad`.

We report a 95% hierarchical paired-bootstrap interval, resampling outer
repetitions and tasks within repetitions. The planned positive result requires
the lower confidence bound of the primary success delta to exceed zero and a
non-negative point estimate for `validation_gated_textgrad - ungated_textgrad`.
Otherwise the experiment is
reported as inconclusive or negative; no post-hoc candidate or split changes
are permitted.

## Reporting

Every train, validation, and test episode; raw candidate output; candidate
filters; validation scores; gate decisions; action traces; and bootstrap inputs
are written beneath the run directory.
