# External Transfer Protocol Results

This run addresses the requested transfer claims:

1. MiniWoB -> WebArena/WorkArena-style browser transfer.
2. tau-bench retail -> tau-bench airline/banking-style tool-policy transfer.
3. SWE-bench dev -> SWE-bench Lite-style coding-policy transfer.

These are **local transfer protocol probes**, not official leaderboard executions. They use the repo's shared external trajectory adapter and deterministic task contracts to test whether source-domain textual policy updates improve held-out target-domain tasks. Official WebArena/WorkArena, tau-bench, and SWE-bench execution remains blocked on backend/runtime requirements listed below.

## Command

```bash
.venv/bin/python -m textgrad_rl.benchmarks.transfer_protocol_suites \
  --output-dir runs/external_transfer_protocols
```

## Results

| Transfer | Method | Source | Target | Episodes | Success | Reward | Invalid | Repeated | Turns | Updates |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Browser | fixed_policy | MiniWoB | WebArena+WorkArena | 6 | 0.000 | 0.250 | 1.000 | 0.167 | 10.17 | 0 |
| Browser | textgrad_rl | MiniWoB | WebArena+WorkArena | 6 | 0.833 | 0.944 | 0.000 | 0.000 | 6.83 | 1 |
| Tool use | fixed_policy | tau-retail | tau-airline+tau-banking | 6 | 0.000 | 0.325 | 0.833 | 0.000 | 10.67 | 0 |
| Tool use | textgrad_rl | tau-retail | tau-airline+tau-banking | 6 | 0.833 | 0.967 | 0.000 | 0.000 | 6.33 | 1 |
| Coding | fixed_policy | SWE-bench-dev | SWE-bench-Lite | 5 | 0.000 | 0.273 | 0.400 | 0.000 | 9.40 | 0 |
| Coding | textgrad_rl | SWE-bench-dev | SWE-bench-Lite | 5 | 0.800 | 0.960 | 0.000 | 0.000 | 5.40 | 1 |

## Learned Transfer Rules

Browser transfer learned from MiniWoB:

- `select_then_submit`
- `avoid_repeated_browser_action`
- `search_before_detail`

Tool-policy transfer learned from retail:

- `verify_identity`
- `check_policy`
- `confirm_final_state`

Coding-policy transfer learned from dev repairs:

- `reproduce_failure`
- `preserve_api`
- `targeted_tests`
- `trace_cross_file_flow`

Each target suite includes one held-out target-specific requirement that is not present in source training, so TextGrad-RL improves strongly but does not reach perfect transfer.

## Official Backend Status

The local machine is not currently configured for official heavyweight benchmark execution:

| Backend | Status |
|---|---|
| WebArena | Missing official site URL env vars: `SHOPPING`, `SHOPPING_ADMIN`, `REDDIT`, `GITLAB`, `MAP`, `WIKIPEDIA`, `HOMEPAGE`. |
| WorkArena | `browsergym-workarena` is not installed and no ServiceNow/WorkArena instance env var is configured. |
| tau-bench | No LLM provider API key is configured for official user-simulator evaluation. |
| SWE-bench | Docker is unavailable and the `swebench` package is not installed; official scoring requires Dockerized repo test execution. |

## Interpretation

These probes are useful for paper development because they test the transfer mechanism in the same shape as the proposed official experiments: train textual policy updates on a source domain, freeze the update, and evaluate on held-out target tasks. They should be described as transfer-protocol probes or pre-official integration checks, not as official WebArena, tau-bench, or SWE-bench results.

The next step for main-conference evidence is to replace each local target contract with the official backend:

- WebArena/WorkArena: configure hosted sites or ServiceNow instance, then run real browser episodes.
- tau-bench: run retail training trajectories and airline/banking evaluation with an LLM user simulator.
- SWE-bench: generate patches on a dev split and score held-out Lite tasks through Docker.

## Artifacts

- `runs/external_transfer_protocols/summary.md`
- `runs/external_transfer_protocols/summary.csv`
- `runs/external_transfer_protocols/official_backend_status.json`
- `runs/external_transfer_protocols/browser_transfer/summary.md`
- `runs/external_transfer_protocols/tau_transfer/summary.md`
- `runs/external_transfer_protocols/swe_transfer/summary.md`
- `runs/external_transfer_protocols/*/fixed_test.jsonl`
- `runs/external_transfer_protocols/*/textgrad_test.jsonl`
