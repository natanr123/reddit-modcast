"""Codex backend: rendering, envelope parsing, and backend dispatch (no live CLI)."""
from __future__ import annotations

import json

import pytest

from modcast import codex_backend as cb


TOOLS = [{
    "name": "compare_removal_rate",
    "description": "compare",
    "strict": True,
    "input_schema": {"type": "object", "properties": {"where_sql": {"type": "string"}},
                     "required": ["where_sql"], "additionalProperties": False},
}]


def fake_run(reply: str):
    def _fake(prompt, model, effort, schema):
        fake_run.last_prompt = prompt
        fake_run.last_schema = schema
        return reply, 0.1
    return _fake


def test_tool_call_envelope_parses(monkeypatch):
    reply = json.dumps({"tool_calls": [{"name": "compare_removal_rate",
                                        "input": {"where_sql": "length(selftext) < 300"}}]})
    monkeypatch.setattr(cb, "_run", fake_run(reply))
    msg = cb.complete("sys", TOOLS, [{"role": "user", "content": "go"}], "m", "low")
    assert msg.stop_reason == "tool_use"
    assert msg.content[0].name == "compare_removal_rate"
    assert msg.content[0].input["where_sql"] == "length(selftext) < 300"
    assert fake_run.last_schema == cb.envelope_schema(TOOLS)


def test_final_text_envelope(monkeypatch):
    monkeypatch.setattr(cb, "_run", fake_run(json.dumps({"final_text": "done"})))
    msg = cb.complete("sys", TOOLS, [{"role": "user", "content": "go"}], "m", "low")
    assert msg.stop_reason == "end_turn"
    assert msg.content[0].text == "done"


def test_non_json_reply_degrades_to_text(monkeypatch):
    monkeypatch.setattr(cb, "_run", fake_run("I forgot the envelope"))
    msg = cb.complete("sys", TOOLS, [{"role": "user", "content": "go"}], "m", "low")
    assert msg.stop_reason == "end_turn"
    assert "forgot" in msg.content[0].text


def test_output_format_passes_user_schema(monkeypatch):
    monkeypatch.setattr(cb, "_run", fake_run('```json\n{"p_removed": 0.4}\n```'))
    fmt = {"type": "json_schema", "schema": {"type": "object", "properties": {}}}
    msg = cb.complete(None, [], [{"role": "user", "content": "estimate"}], "m", "low",
                      output_format=fmt)
    assert fake_run.last_schema == fmt["schema"]
    assert json.loads(msg.content[0].text)["p_removed"] == 0.4  # fences stripped


def test_conversation_rendering_round_trip(monkeypatch):
    monkeypatch.setattr(cb, "_run", fake_run(json.dumps({"final_text": "x"})))
    messages = [
        {"role": "user", "content": "dossier"},
        {"role": "assistant", "content": [cb.ToolUseBlock(id="c1", name="compare_removal_rate",
                                                          input={"where_sql": "over_18"})]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1",
                                      "content": '{"lift": 2.0}', "is_error": False}]},
    ]
    cb.complete("sys", TOOLS, messages, "m", "low")
    p = fake_run.last_prompt
    assert "=== SYSTEM ===" in p and "sys" in p
    assert '<tool_call id=c1 name=compare_removal_rate>{"where_sql": "over_18"}</tool_call>' in p
    assert '<tool_result for=c1>{"lift": 2.0}</tool_result>' in p
    assert p.index("dossier") < p.index("<tool_call") < p.index("<tool_result")


def test_session_dispatches_to_codex(monkeypatch, tmp_path):
    from modcast import llm

    monkeypatch.setattr(llm, "TRAJ_DIR", tmp_path)
    monkeypatch.setattr(cb, "_run", fake_run("hello"))  # no tools -> plain text, no envelope
    s = llm.LLMSession(run_id="t", name="t", backend="codex-cli", model="m")
    resp = s.step("hi")
    assert llm.text_of(resp) == "hello"
    assert s.messages[-1]["role"] == "assistant"
    # trajectory logged the codex model tag
    logged = (tmp_path / "t" / "t.jsonl").read_text()
    assert "codex:m" in logged
