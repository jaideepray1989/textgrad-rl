# TextGrad-RL+ Results

TextGrad-RL+ implements five improvements over the previous TextGrad-RL suite:

1. Multi-candidate textual optimization.
2. Causal credit assignment to environment-specific text variables.
3. Train-replay plus validation scoring.
4. Bootstrap-delta acceptance gating.
5. A learned rule library that is retrieved into final test prompts.

Reproduction command:

```bash
.venv/bin/python -m textgrad_rl.benchmarks.textarena_textgrad_plus \
  --repetitions 3 \
  --train-seeds 5 \
  --val-seeds 5 \
  --test-seeds 10 \
  --output-dir runs/textarena_textgrad_plus
```

Held-out test results use 390 TextArena episodes per method.

| method | reward | reward 95% CI | success | invalid | turns | accepted/repetition | candidates/repetition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_prompt | 0.185 | [0.118, 0.245] | 0.205 | 0.105 | 12.064 | 0.0 | 0.0 |
| scalar_prompt_search | 0.185 | [0.124, 0.245] | 0.205 | 0.105 | 12.064 | 1.0 | 1.0 |
| modular_textgrad | 0.563 | [0.503, 0.623] | 0.615 | 0.000 | 10.708 | 1.0 | 1.0 |
| textgrad_rl_plus | 0.583 | [0.524, 0.641] | 0.615 | 0.000 | 10.156 | 6.0 | 30.0 |

TextGrad-RL+ improves over modular TextGrad-RL by `+0.0205` average reward and `-0.5513` average turns while preserving the same success and invalid-move rates.

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

The most important rejection is `Blackjack-v0`: vanilla modular TextGrad-RL accepts the basic threshold rule and drops held-out Blackjack reward from `0.380` to `0.113`; TextGrad-RL+ rejects that rule and keeps Blackjack at `0.380`.
