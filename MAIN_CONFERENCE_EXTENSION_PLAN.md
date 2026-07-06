# Main-Conference Extension Plan

This update targets the three main weaknesses in the REALM draft:

1. Candidate generator quality and stochastic statistical rigor.
2. True action-probability ratios for PPO-style textual trust regions.
3. External validity beyond TextArena, especially WebArena and SWE-bench.

## 1. Candidate Pool + 30-Seed SLM Rigor

Implemented in `textgrad_rl.benchmarks.textarena_expanded_suites`:

- `--slm-candidate-count N` evaluates a pool of candidate prompt-policy edits instead of a single edit.
- `--slm-candidate-model MODEL` optionally uses a stronger generator model for candidate rules.
- `candidate_decisions.jsonl` logs every candidate's train/validation metrics, gate result, PPO diagnostics, and rank score.
- `update_decisions.jsonl` records the selected candidate per method.
- The selected candidate is the best accepted candidate; if no candidate passes the gate, the method keeps the old text policy.

Recommended long run:

```bash
bash scripts/run_slm_candidate_pool_30seed.sh
```

Equivalent expanded command:

```bash
.venv/bin/python -m textgrad_rl.benchmarks.textarena_expanded_suites \
  --suites puzzle,social,real_slm \
  --slm-methods fixed_prompt_slm,textgrad_rl_no_gate_slm,textgrad_rl_train_val_slm,textgrad_rl_ppo_slm \
  --slm-train-seeds 30 \
  --slm-val-seeds 30 \
  --slm-test-seeds 30 \
  --slm-candidate-count 8 \
  --slm-candidate-model gpt-oss:20b \
  --slm-candidate-temperature 0.2 \
  --output-dir runs/textarena_slm_candidate_pool_30seed \
  --model qwen2.5:3b \
  --temperature 0.7 \
  --timeout 120
```

This is the highest-priority experiment for an extended paper. It directly tests whether PPO-style gating becomes useful when it has enough candidate diversity and enough stochastic samples to distinguish genuine improvements from lucky rollouts.

## 2. True Action Probabilities

Implemented scaffolding:

- `textgrad_rl.benchmarks.action_probability.ActionLogprobScorer`
- `OpenWeightActionLogprobScorer`, an optional Hugging Face causal-LM scorer.
- `action_probability_ratio(...)`, which computes `p_new(action | prompt_new) / p_old(action | prompt_old)` for the same action.
- Optional OpenAI-compatible chat logprob capture via `--slm-request-logprobs`.

The current SLM runner can store generated-action logprob diagnostics when the backend supports chat-completion logprobs. For mathematically cleaner PPO, the next integration step is to score the same sampled action under both old and candidate text prompts using `OpenWeightActionLogprobScorer`. That requires an open-weight actor available through Hugging Face or another scoring backend, not only an Ollama chat-generation endpoint.

Minimal scorer example:

```python
from textgrad_rl.benchmarks.action_probability import OpenWeightActionLogprobScorer, action_probability_ratio

scorer = OpenWeightActionLogprobScorer("Qwen/Qwen2.5-7B-Instruct", device="mps")
ratio = action_probability_ratio(
    scorer,
    old_prompt="TEXT VARIABLES:\n...",
    new_prompt="TEXT VARIABLES:\n...",
    action="[42]",
)
print(ratio.ratio, ratio.clipped_ratio)
```

For the paper, this should be framed as the path from proxy PPO to true text-policy PPO: same frozen model, same candidate action, different text-policy prompts, exact continuation logprobs.

## 3. WebArena and SWE-bench

Implemented scaffolding:

- `textgrad_rl.benchmarks.external_adapter.ExternalAgentEpisode`
- `ExternalAgentStep`
- `gradients_from_external_episodes(...)`

This adapter converts WebArena/SWE-bench style trajectories into the same textual-gradient format used by the TextArena runner. It does not vendor or run those heavyweight benchmarks. A full run should be a separate integration because WebArena requires browser/site infrastructure and SWE-bench requires repository checkout, patch generation, and test execution.

Recommended staged plan:

1. SWE-bench Lite smoke: 10 tasks, fixed frozen actor, collect `ExternalAgentEpisode` JSONL.
2. TextGrad-RL over task-family prompt variables: planning, patch generation, test triage, and submission policy.
3. Train/validation/test split by repository/task ID.
4. Compare fixed prompt, train/val TextGrad-RL, and TextGrad-RL-PPO with exact action logprob ratios if the actor is open-weight.
5. Only then attempt WebArena, where environment setup and browser determinism become the dominant source of noise.

## Current Status

Done:

- Multi-candidate SLM candidate generator path.
- 30-seed runner script.
- Candidate decision logging.
- Optional generated-action logprob capture.
- Exact open-weight action-ratio scorer scaffolding.
- External benchmark trajectory adapter.

Not yet done:

- The long 30-seed SLM run, because it is expected to be materially slower than the current unit/benchmark smoke tests.
- Full true-ratio PPO integration into candidate acceptance.
- Actual WebArena or SWE-bench runs.
