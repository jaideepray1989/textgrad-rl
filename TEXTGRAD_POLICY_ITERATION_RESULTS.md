# TextGrad Policy Iteration Results

This run strengthens the RL component of TextGrad-RL with:

1. Action-level credit assignment from trajectory actions, repeated actions, invalid moves, turn efficiency, and returns.
2. Advantage-weighted textual gradients that target worst negative-advantage actions.
3. Candidate prompt policy search over a text-policy population.
4. Replay buffers for train, replay, validation, action credits, and candidate evaluations.
5. A tabular value critic that estimates environment baselines and candidate advantages.

Reproduction command:

```bash
.venv/bin/python -m textgrad_rl.benchmarks.textarena_policy_iteration \
  --repetitions 3 \
  --train-seeds 5 \
  --val-seeds 5 \
  --test-seeds 10 \
  --output-dir runs/textarena_policy_iteration
```

Held-out test results use 390 TextArena episodes for `textgrad_policy_iteration`.

| method | reward | reward 95% CI | success | invalid | turns | accepted/repetition | candidates/repetition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| textgrad_policy_iteration | 0.583 | [0.524, 0.638] | 0.615 | 0.000 | 10.156 | 6.0 | 40.0 |

The full comparison remains available in `runs/textarena_policy_iteration/metrics_by_run.csv` and `runs/textarena_policy_iteration/bootstrap_cis.csv`. The key mechanism result is that all accepted updates are selected as `advantage_weighted_env_rule` candidates using action-credit evidence and value-critic advantages.

Accepted rule environments in every repetition:

- `FrozenLake-v0`
- `GuessTheNumber-v0`
- `LightsOut-v0`
- `Mastermind-v0`
- `Nim-v0`
- `TowerOfHanoi-v0`

Rejected rule environments in every repetition:

- `Bandit-v0`
- `Blackjack-v0`
- `ConnectFour-v0`
- `ReverseTicTacToe-v0`

Key artifacts:

- `runs/textarena_policy_iteration/methods/textgrad_policy_iteration/rep_*/action_credits.json`
- `runs/textarena_policy_iteration/methods/textgrad_policy_iteration/rep_*/advantage_assignments.json`
- `runs/textarena_policy_iteration/methods/textgrad_policy_iteration/rep_*/replay_buffer.json`
- `runs/textarena_policy_iteration/methods/textgrad_policy_iteration/rep_*/value_critic.json`
- `runs/textarena_policy_iteration/methods/textgrad_policy_iteration/rep_*/candidate_evaluations.jsonl`
