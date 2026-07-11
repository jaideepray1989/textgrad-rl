# qwen2.5:7b Local Text-Game SLM Report

Model: `qwen2.5:7b` through the local Ollama OpenAI-compatible endpoint.
Temperature: `0.7`.

Policies:
- `no_textgrad`: frozen SLM actor with the initial prompt-policy text only.
- `textgrad_rl`: one TextGrad-RL textual rule update selected by a train/validation gate.

Coverage:
- TextArena: 12 SLM-enabled environments: 4 puzzle, 3 social, 5 real-SLM; one train/validation/test seed per environment.
- TextWorldExpress: all 8 local games; one train/validation/test seed per game; 12-step cap.
- TextWorld 24: all 24 generated Microsoft TextWorld games; 3-step cap because longer qwen2.5:7b TextWorld episodes produced long local model calls. Treat this as a shallow interaction probe, not a full solve-rate run.

## Overall Results

| benchmark | policy | problems | episodes | reward | score | success | invalid | parse_fail | repeated | truncated | turns | accepted_updates | gradient_count | max_steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TextArena puzzle | no_textgrad | 4 | 4 | 0.146 | -0.322 | 0.000 | 0.750 | 0.000 | 0.000 | 0.250 | 6.250 | 0 |  | env default |
| TextArena puzzle | textgrad_rl | 4 | 4 | 0.196 | -0.324 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 4.000 | 0 |  | env default |
| TextArena social | no_textgrad | 3 | 3 | -0.333 | -0.583 | 0.000 | 0.333 | 0.000 | 0.000 | 0.000 | 16.667 | 0 |  | env default |
| TextArena social | textgrad_rl | 3 | 3 | -0.333 | -0.583 | 0.000 | 0.333 | 0.000 | 0.000 | 0.000 | 16.667 | 1 |  | env default |
| TextArena real_slm | no_textgrad | 5 | 5 | 0.220 | 0.137 | 0.200 | 0.200 | 0.000 | 0.000 | 0.200 | 6.600 | 0 |  | env default |
| TextArena real_slm | textgrad_rl | 5 | 5 | 0.760 | 0.969 | 0.600 | 0.000 | 0.000 | 0.000 | 0.200 | 8.200 | 1 |  | env default |
| TextWorldExpress 8 | no_textgrad | 8 | 8 | -0.062 | -0.097 | 0.125 | 0.000 | 0.000 | 0.625 |  | 8.500 | 0 | 0 | 12 |
| TextWorldExpress 8 | textgrad_rl | 8 | 8 | -0.078 | -0.113 | 0.125 | 0.000 | 0.000 | 0.625 |  | 8.625 | 0 | 7 | 12 |
| TextWorld 24 | no_textgrad | 24 | 24 | 0.167 | 0.185 | 0.167 | 0.000 | 0.000 | 0.542 |  | 2.792 | 0 | 0 | 3 |
| TextWorld 24 | textgrad_rl | 24 | 24 | 0.167 | 0.189 | 0.167 | 0.000 | 0.000 | 0.500 |  | 2.792 | 0 | 3 | 3 |

## TextGrad-RL Minus No-TextGrad

| benchmark | delta_reward | delta_score | delta_success | delta_invalid | delta_parse_fail | delta_repeated | delta_turns | accepted_updates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TextArena puzzle | 0.050 | -0.001 | 0.000 | 0.250 | 0.000 | 0.000 | -2.250 | 0 |
| TextArena real_slm | 0.540 | 0.832 | 0.400 | -0.200 | 0.000 | 0.000 | 1.600 | 1 |
| TextArena social | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1 |
| TextWorld 24 | 0.000 | 0.004 | 0.000 | 0.000 | 0.000 | -0.042 | 0.000 | 0 |
| TextWorldExpress 8 | -0.016 | -0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.125 | 0 |

## Gate Decisions

| benchmark | policy | accepted | old_score | new_score | gradient_count | note |
| --- | --- | --- | --- | --- | --- | --- |
| TextArena puzzle | textgrad_rl | False | -0.355 | -0.381 |  | train_val |
| TextArena social | textgrad_rl | True | 0.402 | 0.403 |  | train_val |
| TextArena real_slm | textgrad_rl | True | 0.024 | 0.027 |  | train_val |
| TextWorldExpress 8 | no_textgrad | False |  |  | 0 |  |
| TextWorldExpress 8 | textgrad_rl | False | -0.369 | -0.369 | 7 |  |
| TextWorld 24 | no_textgrad | False |  |  | 0 |  |
| TextWorld 24 | textgrad_rl | False | 0.714 | 0.714 | 3 |  |

## Interpretation

- The clearest positive result is TextArena real-SLM: TextGrad-RL improves reward from 0.220 to 0.760, success from 0.200 to 0.600, and removes invalid moves in this one-seed sweep.
- TextArena puzzle shows a small reward increase but worse invalid-action rate, so the update is not a clean win there.
- TextArena social is unchanged: the qwen actor produces valid but non-winning long social trajectories, and the TextGrad rule does not change held-out outcomes.
- TextWorldExpress and TextWorld 24 do not show meaningful improvement from the gated update in this run. Their gates reject the update, so the final policies are effectively the fixed prompt; small score differences come from stochastic qwen rollouts.
- TextWorld 24 should be interpreted cautiously because the run uses a 3-step cap to complete all 24 games locally. A stronger claim needs either a per-call hard timeout/retry wrapper or a faster deterministic local model-serving path for longer episodes.

## Artifacts

- TextArena: `runs/qwen25_7b_full_textgames/textarena/`
- TextWorldExpress/TextWorld: `runs/qwen25_7b_full_textgames/textworld_slm/`
- Consolidated CSVs: `runs/qwen25_7b_full_textgames/report_tables/`
