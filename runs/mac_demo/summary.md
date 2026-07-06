# TextGrad-RL Experiment Summary

- Method: `modular_textgrad`
- Agent: `heuristic`
- Critic: `heuristic`
- Tasks: train=8, val=4, test=4
- Platform: macOS-26.5-arm64-arm-64bit-Mach-O (arm64)

## Final Metrics

| split | success_rate | average_reward | test_pass_rate | invalid_action_rate | average_steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| test | 1.000 | 10.161 | 1.000 | 0.000 | 10.500 |

## Prompt Updates

- Accepted updates: 1
- Rejected updates: 1

## Common Failure Modes

- Reproducibility failure needs explicit validation: 14
- Invalid or forbidden edit attempted: 8
- Patched before inspecting target file: 8
- Schema mismatch not triaged through preprocessing: 4
- Shape mismatch not converted into targeted inspection: 4
- Latency regression from repeated inference work: 2
- Metric regression requires source-level root-cause repair: 2

## Example Textual Gradient

- Target: `patch_planning_prompt`
- Failure mode: Invalid or forbidden edit attempted
- Suggested edit: Add a rule: avoid forbidden shortcuts such as editing tests, metadata, hidden validation, thresholds, data, or expected outputs.

## Final Text Variables

### experiment_planning_prompt

```text
Use the limited step budget to alternate between tests, code inspection, and targeted edits. Avoid repeating the same failing command many times in a row.
```

### log_interpretation_prompt

```text
Summarize the first concrete error in pytest, training, or eval output. Use tracebacks to connect a failure to a source file and line. Treat repeated failures as evidence that the next action should gather more information.

Learned rules:
- for shape mismatch errors, inspect training/preprocessing feature dimensions before editing or rerunning full training.
```

### patch_planning_prompt

```text
Make the smallest source-code change that addresses the observed root cause. Prefer fixes in preprocessing, training, feature, or inference code over broad rewrites. Re-run checks after editing.

Learned rules:
- avoid forbidden shortcuts such as editing tests, metadata, hidden validation, thresholds, data, or expected outputs.
- read a source file before editing it, except for generated scratch artifacts.
```

### triage_prompt

```text
Start from the observable failure and identify the likely subsystem. Prefer reading code near the reported traceback before broad edits. Keep notes about which files explain the failure.

Learned rules:
- for schema mismatch errors, inspect data columns and preprocessing assumptions before editing model code.
```

### validation_prompt

```text
Before submitting, make sure the visible tests and the main training/eval path have completed successfully. Do not submit when the latest output is an error.

Learned rules:
- for reproducibility tasks, set random_state consistently and rerun eval or tests before submit.
```

## Limitations

- The default actor and critic are heuristic stand-ins for frozen small language models.
- The task suite is synthetic and CPU-light, designed for rapid local iteration.
- Local LLM adapters are optional and depend on a user-managed OpenAI-compatible server.
