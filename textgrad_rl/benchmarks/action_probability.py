"""Action-probability helpers for open-weight TextGrad-RL PPO experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


class ActionLogprobScorer(Protocol):
    def logprob(self, prompt: str, action: str) -> float:
        """Return log p(action | prompt) for the frozen actor."""


@dataclass
class ActionProbabilityRatio:
    action: str
    old_logprob: float
    new_logprob: float
    ratio: float
    clipped_ratio: float
    logprob_delta: float


def action_probability_ratio(
    scorer: ActionLogprobScorer,
    old_prompt: str,
    new_prompt: str,
    action: str,
    clip_epsilon: float = 0.2,
) -> ActionProbabilityRatio:
    """Compute a PPO-style probability ratio for the same action under two prompts."""

    old_logprob = scorer.logprob(old_prompt, action)
    new_logprob = scorer.logprob(new_prompt, action)
    logprob_delta = max(-60.0, min(60.0, new_logprob - old_logprob))
    ratio = math.exp(logprob_delta)
    clipped_ratio = min(max(ratio, 1.0 - clip_epsilon), 1.0 + clip_epsilon)
    return ActionProbabilityRatio(
        action=action,
        old_logprob=old_logprob,
        new_logprob=new_logprob,
        ratio=ratio,
        clipped_ratio=clipped_ratio,
        logprob_delta=logprob_delta,
    )


class OpenWeightActionLogprobScorer:
    """Score action continuations with a local Hugging Face causal LM.

    This is optional scaffolding for true textual PPO ratios. It requires
    `transformers` and `torch`, but those packages are intentionally not mandatory
    dependencies for the default CPU-light prototype.
    """

    def __init__(self, model_name_or_path: str, device: str = "cpu") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "OpenWeightActionLogprobScorer requires optional dependencies: transformers and torch."
            ) from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
        self.model.eval()
        self.device = device

    def logprob(self, prompt: str, action: str) -> float:
        full_text = prompt + action
        full = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        prompt_len = int(prompt_ids.shape[-1])
        input_ids = full.input_ids
        if input_ids.shape[-1] <= prompt_len:
            return 0.0
        with self.torch.no_grad():
            logits = self.model(**full).logits
            log_probs = self.torch.log_softmax(logits[:, :-1, :], dim=-1)
        targets = input_ids[:, 1:]
        # Token at original position i is predicted by logits at i - 1.
        start = max(prompt_len - 1, 0)
        end = targets.shape[-1]
        selected = log_probs[:, start:end, :].gather(2, targets[:, start:end].unsqueeze(-1)).squeeze(-1)
        return float(selected.sum().item())
