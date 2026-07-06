"""Optional OpenAI-compatible local LLM actor adapter."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from textgrad_rl.agents.action_parser import parse_action
from textgrad_rl.types import Action, Observation, TextVariable
from textgrad_rl.utils.json_utils import to_jsonable


class LLMActorAgent:
    """Local-LLM actor using an OpenAI-compatible chat/completions endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        timeout: int = 30,
        max_tokens: int = 2000,
    ) -> None:
        self.base_url = (base_url or os.getenv("TEXTGRAD_RL_LLM_BASE_URL") or "http://localhost:11434/v1").rstrip("/")
        self.model = model or os.getenv("TEXTGRAD_RL_LLM_MODEL") or "qwen2.5-coder:3b"
        self.api_key = os.getenv("TEXTGRAD_RL_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "not-needed"
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens

    def act(self, observation: Observation, text_variables: dict[str, TextVariable]) -> Action:
        prompt = self._build_prompt(observation, text_variables)
        try:
            output = self._chat(prompt)
        except Exception as exc:
            return Action(type="noop", reason=f"local LLM request failed: {exc}")
        return parse_action(output)

    def _build_prompt(self, observation: Observation, text_variables: dict[str, TextVariable]) -> str:
        text_modules = {
            name: {"role": var.role_description, "value": var.value}
            for name, var in text_variables.items()
        }
        return (
            "You are a frozen ML-engineering repair agent. Output JSON only with keys "
            "type, path, content, command, reason. Do not edit tests, hidden validation, "
            "metadata, thresholds, or data. Allowed action types are read_file, edit_file, "
            "run_tests, run_training, run_eval, inspect_logs, submit_patch, noop.\n\n"
            f"TEXT VARIABLES:\n{json.dumps(text_modules, indent=2)}\n\n"
            f"OBSERVATION:\n{json.dumps(to_jsonable(observation), indent=2)}"
        )

    def _chat(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return one JSON action object and no prose."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(exc.read().decode("utf-8", errors="replace")) from exc
        return data["choices"][0]["message"]["content"]
