"""Optional local-LLM trajectory critic."""

from __future__ import annotations

import json
import os
import urllib.request

from textgrad_rl.types import TextualGradient, TextVariable, Trajectory
from textgrad_rl.utils.json_utils import to_jsonable


class LLMTrajectoryCritic:
    """Ask a local OpenAI-compatible model for JSON textual gradients."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 60,
        max_tokens: int = 3000,
    ) -> None:
        self.base_url = (base_url or os.getenv("TEXTGRAD_RL_LLM_BASE_URL") or "http://localhost:11434/v1").rstrip("/")
        self.model = model or os.getenv("TEXTGRAD_RL_LLM_MODEL") or "qwen2.5-coder:3b"
        self.api_key = os.getenv("TEXTGRAD_RL_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "not-needed"
        self.timeout = timeout
        self.max_tokens = max_tokens

    def critique(
        self,
        trajectory: Trajectory,
        text_variables: dict[str, TextVariable],
    ) -> list[TextualGradient]:
        prompt = (
            "Return JSON only: a list of textual gradients. Each item must have "
            "target_variable_name, failure_mode, evidence_from_trajectory, gradient_text, "
            "suggested_edit, confidence, forbidden_shortcuts.\n\n"
            f"TEXT VARIABLES:\n{json.dumps(to_jsonable(text_variables), indent=2)}\n\n"
            f"TRAJECTORY:\n{json.dumps(to_jsonable(trajectory), indent=2)}"
        )
        try:
            payload = self._chat(prompt)
            parsed = json.loads(payload)
            if not isinstance(parsed, list):
                return []
            return [TextualGradient(**item) for item in parsed if isinstance(item, dict)]
        except Exception:
            return []

    def _chat(self, prompt: str) -> str:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(
                {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "Output JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": self.max_tokens,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
