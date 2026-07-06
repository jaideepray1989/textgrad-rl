"""Run a real local SLM on TextArena with and without TextGrad prompt updates."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from textgrad_rl.optim.textual_gradient_descent import TextualGradientDescent
from textgrad_rl.types import TextualGradient, TextVariable
from textgrad_rl.utils.json_utils import append_jsonl, write_json
from textgrad_rl.utils.logging import environment_info


ENV_ID = "GuessTheNumber-v0"


@dataclass
class SLMEpisodeRecord:
    env_id: str
    variant: str
    split: str
    seed: int
    model: str
    reward: float
    success: bool
    invalid_move: bool
    turns: int
    reason: str
    actions: list[str]
    raw_outputs: list[str]
    runtime_seconds: float


@dataclass
class ChatCompletionResult:
    text: str
    token_logprobs: list[dict[str, Any]]
    action_logprob: float | None


def first_bracketed_span(text: str) -> tuple[int, int] | None:
    match = re.search(r"\[[^\]]+\]", text)
    if not match:
        return None
    return match.start(), match.end()


def action_span_logprob(text: str, token_logprobs: list[dict[str, Any]]) -> float | None:
    """Approximate generated action logprob from chat-completion token logprobs.

    OpenAI-compatible servers expose generated-token logprobs but do not all expose
    arbitrary continuation scoring. This sums logprobs for tokens overlapping the
    first bracketed action span, which is exact for the generated action span when
    the server returns token text and logprob fields.
    """

    span = first_bracketed_span(text)
    if span is None or not token_logprobs:
        return None
    start, end = span
    offset = 0
    total = 0.0
    used = False
    for item in token_logprobs:
        token = str(item.get("token", ""))
        logprob = item.get("logprob")
        token_start = offset
        token_end = offset + len(token)
        offset = token_end
        if token_end <= start or token_start >= end:
            continue
        if isinstance(logprob, (int, float)):
            total += float(logprob)
            used = True
    return total if used else None


def extract_chat_token_logprobs(choice: dict[str, Any]) -> list[dict[str, Any]]:
    content = choice.get("logprobs", {}).get("content") if isinstance(choice.get("logprobs"), dict) else None
    if not isinstance(content, list):
        return []
    tokens: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        tokens.append(
            {
                "token": item.get("token", ""),
                "logprob": item.get("logprob"),
            }
        )
    return tokens


class OpenAICompatibleChatModel:
    """Tiny OpenAI-compatible chat client for local SLM runtimes."""

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
        request_logprobs: bool = False,
        top_logprobs: int = 0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.request_logprobs = request_logprobs
        self.top_logprobs = top_logprobs
        self.api_key = os.getenv("TEXTGRAD_RL_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "not-needed"

    def complete(self, prompt: str) -> str:
        return self.complete_with_metadata(prompt).text

    def complete_with_metadata(self, prompt: str) -> ChatCompletionResult:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You play TextArena. Return exactly one action in square brackets, "
                        "such as [10]. Do not explain."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.request_logprobs:
            payload["logprobs"] = True
            if self.top_logprobs > 0:
                payload["top_logprobs"] = self.top_logprobs
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(exc.read().decode("utf-8", errors="replace")) from exc
        choice = data["choices"][0]
        text = str(choice["message"]["content"]).strip()
        token_logprobs = extract_chat_token_logprobs(choice)
        return ChatCompletionResult(
            text=text,
            token_logprobs=token_logprobs,
            action_logprob=action_span_logprob(text, token_logprobs),
        )


class SLMGuessTheNumberAgent:
    """Frozen SLM actor controlled only by text variables."""

    def __init__(self, model: OpenAICompatibleChatModel, text_variables: dict[str, TextVariable]) -> None:
        self.model = model
        self.text_variables = text_variables

    def act(self, observation: str) -> tuple[str, str]:
        prompt = self._build_prompt(observation)
        raw = self.model.complete(prompt)
        return normalize_guess_action(raw), raw

    def _build_prompt(self, observation: str) -> str:
        text_modules = "\n\n".join(
            f"{variable.name} ({variable.role_description}):\n{variable.clipped_value()}"
            for variable in self.text_variables.values()
        )
        return (
            f"TEXT VARIABLES:\n{text_modules}\n\n"
            "GAME OBSERVATION:\n"
            f"{observation}\n\n"
            "Return the next legal guess as exactly one bracketed integer from 1 to 20."
        )


def normalize_guess_action(raw_output: str) -> str:
    """Extract or coerce one legal GuessTheNumber action from SLM output."""

    for bracketed in re.findall(r"\[([^\]]+)\]", raw_output):
        numbers = re.findall(r"-?\d+", bracketed)
        if numbers:
            return f"[{clamp_guess(int(numbers[0]))}]"
    numbers = [int(value) for value in re.findall(r"-?\d+", raw_output)]
    for value in numbers:
        return f"[{clamp_guess(value)}]"
    return "[10]"


def clamp_guess(value: int) -> int:
    return max(1, min(20, value))


def initial_slm_variables() -> dict[str, TextVariable]:
    return {
        "guess_the_number_strategy_prompt": TextVariable(
            name="guess_the_number_strategy_prompt",
            value=(
                "Return one legal bracketed number. If uncertain, make conservative guesses "
                "from low to high and avoid prose."
            ),
            role_description="TextArena GuessTheNumber action strategy for a frozen SLM.",
            max_chars=1400,
        )
    }


def gradient_from_records(records: list[SLMEpisodeRecord]) -> list[TextualGradient]:
    failures = [record for record in records if not record.success or record.invalid_move]
    inefficient = [record for record in records if record.turns > 5]
    repeated = sum(has_repeated_guesses(record.actions) for record in records)
    if not failures and not inefficient and not repeated:
        return []
    evidence = (
        f"{len(failures)} of {len(records)} training episodes were not solved cleanly; "
        f"{len(inefficient)} took more than the five guesses needed by binary search; "
        f"{repeated} episodes repeated at least one guess."
    )
    return [
        TextualGradient(
            target_variable_name="guess_the_number_strategy_prompt",
            failure_mode="GuessTheNumber SLM episodes wasted turns or repeated guesses",
            evidence_from_trajectory=evidence,
            gradient_text=(
                "The prompt should force explicit interval tracking from higher/lower hints "
                "instead of unstructured guessing."
            ),
            suggested_edit=(
                "Add a rule: For Guess The Number, track the current lower and upper bounds "
                "from every higher/lower hint, never repeat a previous guess, and choose the "
                "midpoint of the remaining interval."
            ),
            confidence=0.9,
            forbidden_shortcuts=["hardcode the hidden target", "inspect hidden state", "change game rules"],
        )
    ]


def has_repeated_guesses(actions: list[str]) -> bool:
    guesses = [int(match.group(1)) for action in actions if (match := re.search(r"\[(\d+)\]", action))]
    return len(guesses) != len(set(guesses))


def score(records: list[SLMEpisodeRecord]) -> float:
    metrics = summarize(records)
    return (
        3.0 * metrics["average_reward"]
        + 2.0 * metrics["success_rate"]
        - 2.0 * metrics["invalid_move_rate"]
        - 0.01 * metrics["average_turns"]
    )


def variables_changed(old: dict[str, TextVariable], new: dict[str, TextVariable]) -> bool:
    return any(old[name].value != new.get(name, old[name]).value for name in old)


def run_episode(
    variant: str,
    split: str,
    seed: int,
    chat_model: OpenAICompatibleChatModel,
    text_variables: dict[str, TextVariable],
) -> SLMEpisodeRecord:
    try:
        import textarena as ta
    except ImportError as exc:
        raise SystemExit("TextArena is not installed. Run: python -m pip install textarena") from exc

    env = ta.make(ENV_ID)
    env.reset(num_players=1, seed=seed)
    agent = SLMGuessTheNumberAgent(chat_model, text_variables)
    actions: list[str] = []
    raw_outputs: list[str] = []
    done = False
    turns = 0
    start = time.perf_counter()
    while not done:
        _, observation = env.get_observation()
        if not isinstance(observation, str):
            observation = "\n".join(str(item) for item in observation)
        try:
            action, raw = agent.act(observation)
        except Exception as exc:
            raw = f"REQUEST_FAILED: {exc}"
            action = "[10]"
        actions.append(action)
        raw_outputs.append(raw)
        done, _ = env.step(action)
        turns += 1
        if turns > 12:
            raise RuntimeError("GuessTheNumber exceeded expected turn budget")
    rewards, game_info = env.close()
    info = game_info.get(0, {})
    reward = float(rewards.get(0, 0.0))
    return SLMEpisodeRecord(
        env_id=ENV_ID,
        variant=variant,
        split=split,
        seed=seed,
        model=chat_model.model,
        reward=reward,
        success=reward >= 1.0,
        invalid_move=bool(info.get("invalid_move", False)),
        turns=turns,
        reason=str(info.get("reason", "")),
        actions=actions,
        raw_outputs=raw_outputs,
        runtime_seconds=time.perf_counter() - start,
    )


def run_records(
    variant: str,
    split: str,
    count: int,
    seed: int,
    chat_model: OpenAICompatibleChatModel,
    text_variables: dict[str, TextVariable],
    path: Path,
) -> list[SLMEpisodeRecord]:
    if path.exists():
        path.unlink()
    records = []
    for idx in range(count):
        record = run_episode(variant, split, seed + idx, chat_model, text_variables)
        records.append(record)
        append_jsonl(path, record)
    return records


def summarize(records: list[SLMEpisodeRecord]) -> dict[str, Any]:
    n = len(records)
    return {
        "episodes": n,
        "average_reward": sum(record.reward for record in records) / n if n else 0.0,
        "success_rate": sum(record.success for record in records) / n if n else 0.0,
        "invalid_move_rate": sum(record.invalid_move for record in records) / n if n else 0.0,
        "average_turns": sum(record.turns for record in records) / n if n else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def row_for(variant: str, split: str, records: list[SLMEpisodeRecord], accepted: bool) -> dict[str, Any]:
    return {"variant": variant, "split": split, "accepted": accepted, **summarize(records)}


def write_summary(
    path: Path,
    model: str,
    base_url: str,
    rows: list[dict[str, Any]],
    accepted: bool,
    gradients: list[TextualGradient],
) -> None:
    by_key = {(row["variant"], row["split"]): row for row in rows}
    fixed = by_key.get(("no_textgrad", "test"), {})
    textgrad = by_key.get(("textgrad_rl", "test"), {})
    lines = [
        "# TextArena Real SLM Comparison",
        "",
        f"- Environment: {ENV_ID}",
        f"- Model: {model}",
        f"- Base URL: {base_url}",
        f"- TextGrad update accepted: {accepted}",
        "",
        "## Held-Out Test",
        "",
        "| variant | episodes | avg_reward | success | invalid | avg_turns |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, row in [("no_textgrad", fixed), ("textgrad_rl", textgrad)]:
        if row:
            lines.append(
                "| {variant} | {episodes} | {average_reward:.3f} | {success_rate:.3f} | "
                "{invalid_move_rate:.3f} | {average_turns:.3f} |".format(**row)
            )
    if fixed and textgrad:
        lines.extend(
            [
                "",
                "## Delta",
                "",
                f"- Average reward: {textgrad['average_reward'] - fixed['average_reward']:+.3f}",
                f"- Success rate: {textgrad['success_rate'] - fixed['success_rate']:+.3f}",
                f"- Invalid move rate: {textgrad['invalid_move_rate'] - fixed['invalid_move_rate']:+.3f}",
            ]
        )
    lines.extend(["", "## Textual Gradients", ""])
    if gradients:
        for gradient in gradients:
            lines.append(f"- {gradient.failure_mode}: {gradient.suggested_edit}")
    else:
        lines.append("- No gradient emitted; training episodes were already clean.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_slm_comparison(
    base_url: str,
    model: str,
    train_seeds: int,
    val_seeds: int,
    test_seeds: int,
    seed: int,
    output_dir: Path,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> Path:
    try:
        import textarena as ta
        import textarena.api as ta_api
    except ImportError as exc:
        raise SystemExit("TextArena is not installed. Run: python -m pip install textarena") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    games_dir = output_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    chat_model = OpenAICompatibleChatModel(base_url, model, temperature, max_tokens, timeout)
    variables = initial_slm_variables()
    write_json(
        output_dir / "config.json",
        {
            "env_id": ENV_ID,
            "base_url": base_url,
            "model": model,
            "train_seeds": train_seeds,
            "val_seeds": val_seeds,
            "test_seeds": test_seeds,
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
        },
    )
    write_json(
        output_dir / "environment_info.json",
        {
            **environment_info(),
            "textarena_version": getattr(ta, "__version__", "unknown"),
            "registered_env_count": len(getattr(ta_api, "ENV_REGISTRY", {})),
        },
    )
    write_json(output_dir / "initial_text_variables.json", variables)

    rows: list[dict[str, Any]] = []
    no_textgrad_test = run_records(
        "no_textgrad",
        "test",
        test_seeds,
        seed + 2000,
        chat_model,
        variables,
        games_dir / "no_textgrad_test.jsonl",
    )
    rows.append(row_for("no_textgrad", "test", no_textgrad_test, accepted=False))

    train_records = run_records(
        "textgrad_rl",
        "train",
        train_seeds,
        seed,
        chat_model,
        variables,
        games_dir / "textgrad_train.jsonl",
    )
    rows.append(row_for("textgrad_rl", "train", train_records, accepted=False))
    gradients = gradient_from_records(train_records)
    write_json(output_dir / "gradients.json", gradients)
    candidate = TextualGradientDescent(max_prompt_chars=1400, max_rules_per_step=4).step(
        variables,
        gradients,
        constraints=["Do not hardcode the hidden target", "Do not inspect hidden state", "Do not change game rules"],
    )

    val_old = run_records(
        "textgrad_rl",
        "val_old",
        val_seeds,
        seed + 1000,
        chat_model,
        variables,
        games_dir / "textgrad_val_old.jsonl",
    )
    val_candidate = run_records(
        "textgrad_rl",
        "val_candidate",
        val_seeds,
        seed + 1000,
        chat_model,
        candidate,
        games_dir / "textgrad_val_candidate.jsonl",
    )
    changed = variables_changed(variables, candidate)
    accepted = changed and score(val_candidate) >= score(val_old)
    if accepted:
        variables = candidate
    write_json(
        output_dir / ("accepted_updates.json" if accepted else "rejected_updates.json"),
        {
            "accepted": accepted,
            "changed": changed,
            "old_score": score(val_old),
            "new_score": score(val_candidate),
            "old_metrics": summarize(val_old),
            "new_metrics": summarize(val_candidate),
        },
    )
    rows.append(row_for("textgrad_rl", "val_old", val_old, accepted=False))
    rows.append(row_for("textgrad_rl", "val_candidate", val_candidate, accepted=accepted))

    textgrad_test = run_records(
        "textgrad_rl",
        "test",
        test_seeds,
        seed + 2000,
        chat_model,
        variables,
        games_dir / "textgrad_test.jsonl",
    )
    rows.append(row_for("textgrad_rl", "test", textgrad_test, accepted=accepted))

    write_csv(output_dir / "metrics.csv", rows)
    write_json(output_dir / "final_text_variables.json", variables)
    write_summary(output_dir / "summary.md", model, base_url, rows, accepted, gradients)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare real local SLM TextArena prompting with TextGrad-RL.")
    parser.add_argument("--base-url", default=os.getenv("TEXTGRAD_RL_LLM_BASE_URL", "http://localhost:11434/v1"))
    parser.add_argument("--model", default=os.getenv("TEXTGRAD_RL_LLM_MODEL", "qwen2.5:3b"))
    parser.add_argument("--train-seeds", type=int, default=3)
    parser.add_argument("--val-seeds", type=int, default=3)
    parser.add_argument("--test-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7300)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--output-dir", default="runs/textarena_slm_qwen25_3b_guess_number")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = run_slm_comparison(
        base_url=args.base_url,
        model=args.model,
        train_seeds=args.train_seeds,
        val_seeds=args.val_seeds,
        test_seeds=args.test_seeds,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    print(f"TextArena SLM comparison artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
