"""Anthropic client wrapper with trajectory logging.

Every model interaction in ModCast goes through `LLMSession` so that the
hackathon deliverable — full agent trajectories — falls out for free as
JSONL under results/trajectories/.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic

from modcast.config import RESULTS_DIR, LLM_BACKEND, LLM_EFFORT, LLM_MODEL_NAME

DEFAULT_MODEL = LLM_MODEL_NAME
TRAJ_DIR = RESULTS_DIR / "trajectories"


@dataclass
class LLMSession:
    """One logical conversation with the model, logged to a JSONL trajectory.

    `backend` picks the engine: "anthropic" (metered API) or "codex-cli"
    (flat-rate `codex exec` emulating the same tool-use surface). Defaults
    come from LLM_MODEL in .env; see modcast.config.
    """

    run_id: str
    name: str
    model: str = DEFAULT_MODEL
    effort: str | None = None          # None -> LLM_EFFORT from config
    system: str | None = None
    tools: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    backend: str = LLM_BACKEND
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def __post_init__(self) -> None:
        if self.effort is None:
            self.effort = LLM_EFFORT
        if self.backend == "anthropic":
            self.client = anthropic.Anthropic()
        self._traj_path = TRAJ_DIR / self.run_id / f"{self.name}.jsonl"
        self._traj_path.parent.mkdir(parents=True, exist_ok=True)

    # -- trajectory -------------------------------------------------------
    def _log(self, event: str, payload: Any) -> None:
        rec = {"ts": time.time(), "event": event, "payload": payload}
        with self._traj_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    # -- calls ------------------------------------------------------------
    def step(
        self,
        user_content: Any | None = None,
        max_tokens: int = 16000,
        output_format: dict | None = None,
    ):
        """Append optional user content and run one model turn (any backend)."""
        if user_content is not None:
            self.messages.append({"role": "user", "content": user_content})
            self._log("user", user_content)
        if self.backend == "codex-cli":
            from modcast import codex_backend

            response = codex_backend.complete(
                system=self.system, tools=self.tools, messages=self.messages,
                model=self.model, effort=self.effort, output_format=output_format,
            )
        else:
            output_config: dict[str, Any] = {"effort": self.effort}
            if output_format is not None:
                output_config["format"] = output_format
            kwargs: dict[str, Any] = dict(
                model=self.model,
                max_tokens=max_tokens,
                messages=self.messages,
                output_config=output_config,
                cache_control={"type": "ephemeral"},  # auto-cache: tool loops resend history every turn
            )
            if self.system:
                kwargs["system"] = self.system
            if self.tools:
                kwargs["tools"] = self.tools
            response = self._create_with_retry(**kwargs)
        u = response.usage
        self.total_input_tokens += u.input_tokens
        self.total_output_tokens += u.output_tokens
        self.messages.append({"role": "assistant", "content": response.content})
        self._log("assistant", [b.to_dict() for b in response.content])
        self._log("usage", {"stop_reason": response.stop_reason,
                            "model": response.model,
                            "input_tokens": u.input_tokens,
                            "output_tokens": u.output_tokens,
                            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
                            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0})
        return response

    def tool_results(self, results: list[dict]) -> None:
        """Queue tool_result blocks (all results for a turn in ONE user message)."""
        self.messages.append({"role": "user", "content": results})
        self._log("tool_results", results)

    def _create_with_retry(self, max_attempts: int = 5, **kwargs) -> anthropic.types.Message:
        delay = 2.0
        for attempt in range(max_attempts):
            try:
                return self.client.messages.create(**kwargs)
            except anthropic.RateLimitError as e:
                wait = float(e.response.headers.get("retry-after", delay))
                self._log("rate_limited", {"attempt": attempt, "wait": wait})
                time.sleep(wait)
            except anthropic.APIStatusError as e:
                if e.status_code < 500:
                    raise
                self._log("server_error", {"attempt": attempt, "status": e.status_code})
                time.sleep(delay)
            except anthropic.APIConnectionError:
                self._log("connection_error", {"attempt": attempt})
                time.sleep(delay)
            delay = min(delay * 2, 60)
        raise RuntimeError(f"LLM call failed after {max_attempts} attempts")


def new_run_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def text_of(response) -> str:
    return "".join(b.text for b in response.content if b.type == "text")


def tool_uses(response) -> list[Any]:
    return [b for b in response.content if b.type == "tool_use"]
