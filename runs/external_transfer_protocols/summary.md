# External Transfer Protocol Results

These are local, reproducible transfer probes for the requested source->target benchmark protocols. They are not official WebArena/WorkArena, tau-bench, or SWE-bench leaderboard runs.

## Transfer Results

| Transfer | Method | Source | Target | Episodes | Success | Reward | Invalid | Repeated | Turns | Updates |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| browser_transfer | fixed_policy | MiniWoB | WebArena+WorkArena | 6 | 0.000 | 0.250 | 1.000 | 0.167 | 10.17 | 0 |
| browser_transfer | textgrad_rl | MiniWoB | WebArena+WorkArena | 6 | 0.833 | 0.944 | 0.000 | 0.000 | 6.83 | 1 |
| tau_transfer | fixed_policy | tau-retail | tau-airline+tau-banking | 6 | 0.000 | 0.325 | 0.833 | 0.000 | 10.67 | 0 |
| tau_transfer | textgrad_rl | tau-retail | tau-airline+tau-banking | 6 | 0.833 | 0.967 | 0.000 | 0.000 | 6.33 | 1 |
| swe_transfer | fixed_policy | SWE-bench-dev | SWE-bench-Lite | 5 | 0.000 | 0.273 | 0.400 | 0.000 | 9.40 | 0 |
| swe_transfer | textgrad_rl | SWE-bench-dev | SWE-bench-Lite | 5 | 0.800 | 0.960 | 0.000 | 0.000 | 5.40 | 1 |

## Official Backend Status

- `webarena`: `{'can_run_official_backend': False, 'missing_url_vars': ['SHOPPING', 'SHOPPING_ADMIN', 'REDDIT', 'GITLAB', 'MAP', 'WIKIPEDIA', 'HOMEPAGE']}`
- `workarena`: `{'browsergym_workarena_installed': False, 'has_servicenow_instance': False, 'missing_instance_vars': ['SNOW_INSTANCE_URL', 'SERVICENOW_INSTANCE_URL', 'WORKARENA_INSTANCE_URL']}`
- `tau_bench`: `{'has_llm_api_key': False, 'note': 'Official tau2/tau3 evaluation usually requires an LLM user simulator or provider key.'}`
- `swe_bench`: `{'docker_available': False, 'swebench_package_available': False, 'note': 'Official SWE-bench scoring applies generated patches in Docker and runs repository tests.'}`
