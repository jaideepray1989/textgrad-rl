# WebArena Small-Subset Benchmark Status

Status: blocked
Backend: official
Tasks selected: 20
Methods: fixed_actor, textgrad_rl, textgrad_rl_ppo

## Task Mix

Sites: {"gitlab": 2, "gitlab+reddit": 2, "gitlab+wikipedia": 2, "map": 2, "map+shopping_admin": 2, "map+wikipedia": 1, "reddit": 2, "reddit+gitlab": 2, "shopping": 2, "shopping+reddit": 1, "shopping_admin": 1, "wikipedia+map": 1}
Evaluation types: {"program_html": 9, "string_match": 9, "url_match": 5}

## Preflight

### Missing commands
- docker

### Missing imports
- beartype
- gymnasium
- playwright
- tiktoken

### Missing environment variables
- GITLAB
- HOMEPAGE
- MAP
- REDDIT
- SHOPPING
- SHOPPING_ADMIN
- WIKIPEDIA

### Missing generated configs
- /tmp/webarena_inspect/config_files/0.json
- /tmp/webarena_inspect/config_files/21.json
- /tmp/webarena_inspect/config_files/22.json
- /tmp/webarena_inspect/config_files/265.json
- /tmp/webarena_inspect/config_files/27.json
- /tmp/webarena_inspect/config_files/28.json
- /tmp/webarena_inspect/config_files/44.json
- /tmp/webarena_inspect/config_files/45.json
- /tmp/webarena_inspect/config_files/552.json
- /tmp/webarena_inspect/config_files/553.json

### Missing auth files
- /tmp/webarena_inspect/.auth/gitlab.reddit_state.json
- /tmp/webarena_inspect/.auth/gitlab_state.json
- /tmp/webarena_inspect/.auth/reddit_state.json
- /tmp/webarena_inspect/.auth/shopping_admin_state.json
- /tmp/webarena_inspect/.auth/shopping_state.json

### Notes
- Docker is required to self-host WebArena locally when remote site URLs are not set.
- Preflight failed; no task scores were produced.
