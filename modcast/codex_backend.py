"""Codex CLI backend: run `codex exec` as a pure inference engine.

The codex CLI is billed through a flat-rate subscription, so test runs cost
nothing; this module emulates the small slice of the Anthropic Messages
surface that LLMSession uses (content blocks, tool_use loop, stop_reason,
usage), which makes the two backends interchangeable behind
`modcast.llm.LLMSession`.

How the emulation works per turn:
- The whole conversation (system, tool schemas, prior turns, tool results) is
  rendered into one prompt; codex runs stateless with a read-only sandbox and
  is told to act as a model, not an agent (no shell, no file exploration).
- The final message is constrained with `--output-schema` to an envelope:
  {"tool_calls": [{name, input}, ...]} to call tools, or {"final_text": ...}.
- The envelope parses into duck-typed blocks the agent loop already handles.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from modcast.config import PROJECT_ROOT

CODEX_BIN = os.environ.get("CODEX_BIN", "codex")
CALL_TIMEOUT_S = int(os.environ.get("CODEX_TIMEOUT_S", "600"))

def envelope_schema(tools: list[dict]) -> dict:
    """Strict envelope embedding each tool's real input schema.

    OpenAI structured outputs demand additionalProperties:false and a full
    `required` list on EVERY object, so a free-form {"type":"object"} input is
    rejected; embedding the per-tool schemas both satisfies that and gives the
    same mechanical validation of tool inputs the Anthropic strict mode has.
    """
    variants = [
        {
            "type": "object",
            "properties": {"name": {"type": "string", "enum": [t["name"]]},
                           "input": t["input_schema"]},
            "required": ["name", "input"],
            "additionalProperties": False,
        }
        for t in tools
    ]
    return {
        "type": "object",
        "properties": {
            "tool_calls": {"type": "array", "items": {"anyOf": variants}},
            "final_text": {"type": "string"},
        },
        "required": ["tool_calls", "final_text"],
        "additionalProperties": False,
    }

INFERENCE_NOTE = """[Runner note — you are being used as a pure inference engine inside another
program's agent loop, NOT as a coding agent. Do not run shell commands, do not read or write
files, do not explore the repository. Work only from the conversation below and answer directly.]
"""

ENVELOPE_NOTE = """RESPONSE FORMAT: your final message must be a single JSON object with BOTH keys:
- "tool_calls": a list of {"name", "input"} objects to invoke the tools defined above
  (several calls at once are allowed and encouraged when independent), and
- "final_text": your answer text.
Populate exactly ONE of them: while investigating, fill tool_calls and set final_text to "";
when you are done (or need no tools), set tool_calls to [] and give final_text.
Tool results will arrive in a follow-up turn."""


# -- duck-typed response objects (mirror the anthropic SDK surface we use) ----

@dataclass
class TextBlock:
    text: str
    type: str = "text"

    def to_dict(self) -> dict:
        return {"type": "text", "text": self.text}


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"

    def to_dict(self) -> dict:
        return {"type": "tool_use", "id": self.id, "name": self.name, "input": self.input}


@dataclass
class CodexUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class CodexMessage:
    content: list
    stop_reason: str
    model: str
    usage: CodexUsage = field(default_factory=CodexUsage)
    duration_s: float = 0.0


# -- rendering ----------------------------------------------------------------

def _block_dict(block: Any) -> dict:
    if isinstance(block, dict):
        return block
    if hasattr(block, "to_dict"):
        return block.to_dict()
    raise TypeError(f"cannot render block {block!r}")


def render_prompt(
    system: str | None,
    tools: list[dict],
    messages: list[dict],
    output_format: dict | None,
) -> str:
    parts = [INFERENCE_NOTE]
    if system:
        parts += ["=== SYSTEM ===", system]
    if tools:
        specs = [
            {"name": t["name"], "description": t.get("description", ""), "input_schema": t["input_schema"]}
            for t in tools
        ]
        parts += ["=== AVAILABLE TOOLS ===", json.dumps(specs, indent=1), ENVELOPE_NOTE]
    elif output_format is not None:
        parts += [
            "RESPONSE FORMAT: your final message must be a single JSON object matching the "
            "response schema (no prose, no code fences)."
        ]
    parts.append("=== CONVERSATION ===")
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            parts.append(f"[{msg['role'].upper()}]\n{content}")
            continue
        lines = [f"[{msg['role'].upper()}]"]
        for raw in content:
            b = _block_dict(raw)
            if b["type"] == "text":
                lines.append(b["text"])
            elif b["type"] == "tool_use":
                lines.append(
                    f"<tool_call id={b['id']} name={b['name']}>{json.dumps(b['input'])}</tool_call>"
                )
            elif b["type"] == "tool_result":
                err = " is_error=true" if b.get("is_error") else ""
                lines.append(
                    f"<tool_result for={b['tool_use_id']}{err}>{b.get('content', '')}</tool_result>"
                )
            elif b["type"] == "thinking":
                continue  # never replay reasoning
            else:
                lines.append(json.dumps(b))
        parts.append("\n".join(lines))
    parts.append("[ASSISTANT] (respond now)")
    return "\n\n".join(parts)


# -- execution ----------------------------------------------------------------

class CodexError(RuntimeError):
    pass


def _run(prompt: str, model: str, effort: str, schema: dict | None) -> tuple[str, float]:
    if shutil.which(CODEX_BIN) is None:
        raise CodexError(f"codex CLI not found ({CODEX_BIN!r}); install/login it first.")
    with tempfile.TemporaryDirectory(prefix="modcast-codex-") as td:
        out_path = Path(td) / "last_message.txt"
        argv = [
            CODEX_BIN, "exec",
            "--skip-git-repo-check", "-C", str(PROJECT_ROOT),
            "--sandbox", "read-only",
            "--ephemeral",
            "--color", "never",
            "-c", f'model_reasoning_effort="{effort}"',
            "-o", str(out_path),
        ]
        if model:
            argv += ["-m", model]
        if schema is not None:
            schema_path = Path(td) / "schema.json"
            schema_path.write_text(json.dumps(schema))
            argv += ["--output-schema", str(schema_path)]
        argv.append("-")
        t0 = time.time()
        proc = subprocess.run(
            argv, input=prompt.encode(), capture_output=True, timeout=CALL_TIMEOUT_S
        )
        duration = time.time() - t0
        if not out_path.exists():
            tail = (proc.stdout + b"\n" + proc.stderr).decode(errors="replace")[-2000:]
            raise CodexError(
                f"codex exec produced no final message (exit {proc.returncode}). Tail:\n{tail}"
            )
        return out_path.read_text().strip(), duration


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.rstrip().endswith("```"):
            t = t.rstrip()[: -3]
    return t.strip()


def complete(
    system: str | None,
    tools: list[dict],
    messages: list[dict],
    model: str,
    effort: str,
    output_format: dict | None = None,
    max_attempts: int = 3,
) -> CodexMessage:
    """One emulated Messages turn through `codex exec`."""
    schema = None
    if tools:
        schema = envelope_schema(tools)
    elif output_format is not None:
        schema = output_format.get("schema", output_format)
    prompt = render_prompt(system, tools, messages, output_format)

    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            text, duration = _run(prompt, model, effort, schema)
            break
        except (CodexError, subprocess.TimeoutExpired) as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    else:
        raise CodexError(f"codex failed after {max_attempts} attempts: {last_err}")

    tag = f"codex:{model}" if model else "codex:default"
    if not tools:
        return CodexMessage(
            content=[TextBlock(_strip_fences(text) if output_format else text)],
            stop_reason="end_turn", model=tag, duration_s=duration,
        )
    try:
        envelope = json.loads(_strip_fences(text))
    except json.JSONDecodeError:
        # model ignored the envelope; surface its text so the loop can nudge
        return CodexMessage(content=[TextBlock(text)], stop_reason="end_turn",
                            model=tag, duration_s=duration)
    calls = envelope.get("tool_calls") or []
    if calls:
        blocks = [
            ToolUseBlock(id=f"codex_call_{int(time.time() * 1000)}_{i}", name=c["name"],
                         input=c.get("input", {}))
            for i, c in enumerate(calls)
        ]
        return CodexMessage(content=blocks, stop_reason="tool_use", model=tag, duration_s=duration)
    return CodexMessage(content=[TextBlock(envelope.get("final_text", ""))],
                        stop_reason="end_turn", model=tag, duration_s=duration)
