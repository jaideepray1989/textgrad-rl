from textgrad_rl.agents.action_parser import parse_action


def test_parse_strict_json_action():
    action = parse_action('{"type": "read_file", "path": "train.py", "reason": "inspect"}')
    assert action.type == "read_file"
    assert action.path == "train.py"


def test_parse_fenced_json_action():
    action = parse_action('```json\n{"type": "run_tests", "reason": "check"}\n```')
    assert action.type == "run_tests"


def test_parse_invalid_action_returns_noop():
    action = parse_action("not json")
    assert action.type == "noop"
    assert "invalid" in (action.reason or "")

