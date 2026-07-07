from argparse import Namespace
from pathlib import Path

from textgrad_rl.benchmarks.webarena_subset import (
    WebArenaPreflight,
    build_method_configs,
    check_webarena_preflight,
    load_raw_tasks,
    select_task_subset,
    write_blocked_artifacts,
)
from textgrad_rl.utils.json_utils import read_json


def _raw_task(task_id: int, site: str) -> dict:
    return {
        "task_id": task_id,
        "sites": [site],
        "intent": f"Do task {task_id}",
        "start_url": f"__{site.upper()}__",
        "require_login": True,
        "storage_state": f"./.auth/{site}_state.json",
        "eval": {"eval_types": ["string_match"]},
        "intent_template_id": 100 + task_id,
    }


def test_select_task_subset_balances_sites() -> None:
    raw_tasks = [
        _raw_task(0, "shopping_admin"),
        _raw_task(1, "shopping_admin"),
        _raw_task(2, "gitlab"),
        _raw_task(3, "gitlab"),
        _raw_task(4, "reddit"),
    ]

    selected = select_task_subset(raw_tasks, task_count=4)

    assert [task.task_id for task in selected] == ["2", "4", "0", "3"]
    assert {task.sites[0] for task in selected} == {"shopping_admin", "gitlab", "reddit"}


def test_preflight_reports_missing_infrastructure(tmp_path: Path) -> None:
    root = tmp_path / "webarena"
    root.mkdir()
    task_config = root / "config_files" / "test.raw.json"
    task_config.parent.mkdir()
    task_config.write_text("[]\n", encoding="utf-8")
    selected = select_task_subset([_raw_task(0, "shopping_admin")], task_count=1)

    preflight = check_webarena_preflight(
        backend="official",
        webarena_root=root,
        task_config=task_config,
        selected_tasks=selected,
        env={},
        command_resolver=lambda _name: None,
        import_checker=lambda _name: False,
    )

    assert not preflight.ok
    assert "docker" in preflight.missing_commands
    assert "SHOPPING_ADMIN" in preflight.missing_env_vars
    assert "playwright" in preflight.missing_imports
    assert preflight.missing_generated_configs
    assert preflight.missing_auth_files


def test_write_blocked_artifacts(tmp_path: Path) -> None:
    tasks = select_task_subset([_raw_task(0, "shopping_admin"), _raw_task(1, "gitlab")], task_count=2)
    args = Namespace(
        task_count=2,
        backend="official",
        llm_base_url="http://localhost:11434/v1",
        model="gpt-oss:20b",
        temperature=0.7,
        llm_max_tokens=384,
        max_steps=30,
        train_tasks=1,
        val_tasks=1,
    )
    preflight = WebArenaPreflight(
        ok=False,
        backend="official",
        webarena_root="/missing/webarena",
        task_config="/missing/webarena/config_files/test.raw.json",
        selected_task_count=2,
        missing_commands=["docker"],
        missing_imports=["playwright"],
        missing_env_vars=["SHOPPING"],
        missing_generated_configs=[],
        missing_auth_files=[],
        notes=["Preflight failed; no task scores were produced."],
    )

    write_blocked_artifacts(
        tmp_path,
        args=args,
        selected_tasks=tasks,
        method_configs=build_method_configs(["fixed_actor", "textgrad_rl", "textgrad_rl_ppo"]),
        preflight=preflight,
    )

    assert read_json(tmp_path / "task_subset_summary.json")["task_count"] == 2
    assert "fixed_actor" in (tmp_path / "summary.csv").read_text(encoding="utf-8")
    assert "not_run_preflight_failed" in (tmp_path / "episodes.jsonl").read_text(encoding="utf-8")


def test_load_raw_tasks_rejects_non_list(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"task_id": 1}\n', encoding="utf-8")

    try:
        load_raw_tasks(path)
    except ValueError as exc:
        assert "Expected WebArena raw task list" in str(exc)
    else:
        raise AssertionError("load_raw_tasks should reject non-list JSON")
