from textgrad_rl.envs.task_factory import FAMILIES, build_task_splits, create_task_spec


def test_task_factory_builds_required_families():
    for family in FAMILIES:
        spec = create_task_spec(family, 9)
        assert spec.files
        assert "hidden_validation.py" in spec.hidden_files
        assert spec.metric_name


def test_task_splits_have_requested_counts():
    train, val, test = build_task_splits(3, 2, 1, seed=4)
    assert len(train) == 3
    assert len(val) == 2
    assert len(test) == 1
    assert {task.split for task in train} == {"train"}

