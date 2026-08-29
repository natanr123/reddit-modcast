"""Offline rulebook induction: the agent's verified memory.

Per subreddit, the model studies real removed/survived examples and the
published rules, proposes candidate removal drivers as where_sql predicates,
and tests them with compare_removal_rate. Nothing it claims is trusted:
every finalized entry is RE-RUN by code, and only entries whose measured
effect passes thresholds are written into rulebooks/{sub}.md with their
actual rates. Prediction-time agents read this file; playbook-on vs
playbook-off is a clean changelog ablation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import duckdb

from modcast import stats as S
from modcast.config import RULEBOOK_DIR, RANDOM_SEED, index_window
from modcast.llm import LLMSession, tool_uses
from modcast.subrules import rules_digest

MIN_GROUP_N = 30
MIN_EFFECT = 0.08  # absolute percentage-point gap to keep a rule

SYSTEM = """You are building a moderation rulebook for one subreddit from real
labeled data. Propose hypotheses about what gets posts REMOVED BY MODERATION
here, and test each with compare_removal_rate. Good hypotheses come from: the
published rules (automod usually enforces format/flair/length rules), patterns
you notice in the example posts, and general automod practice (required title
tags, minimum length, link policies, flair requirements).

Test 8-15 hypotheses, including some you expect to be FALSE (knowing what
does NOT matter here is valuable). Then call finalize_rulebook with the
entries worth keeping. Every entry is re-verified by code against the full
corpus — entries that fail verification are dropped silently, so prefer
predicates you have already seen produce a real gap.

where_sql mini-language: columns title, selftext, is_self, over_18,
link_flair_text, author_flair_text, created_utc, text_available; functions
length, lower, upper, regexp_matches, strftime, epoch; operators AND OR NOT
LIKE ILIKE = != < > <= >= IS NULL."""

TOOLS = [
    {
        "name": "compare_removal_rate",
        "description": "Compare removal rate where the predicate is TRUE vs FALSE over the full labeled corpus for this subreddit.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "where_sql": {"type": "string"},
                "hypothesis": {"type": "string"},
            },
            "required": ["where_sql", "hypothesis"],
            "additionalProperties": False,
        },
    },
    {
        "name": "finalize_rulebook",
        "description": "Submit the rulebook entries. Call once, when done testing.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "entries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "rule": {"type": "string", "description": "Human-readable statement of the driver."},
                            "where_sql": {"type": "string", "description": "Predicate identifying the risky trait (TRUE = has the trait)."},
                            "maps_to": {"type": ["string", "null"], "description": "Published rule this reflects, if any."},
                        },
                        "required": ["rule", "where_sql", "maps_to"],
                        "additionalProperties": False,
                    },
                },
                "observations": {"type": "string", "description": "2-4 sentences: what does NOT matter here, and anything surprising."},
            },
            "required": ["entries", "observations"],
            "additionalProperties": False,
        },
    },
]


@dataclass
class InductionResult:
    subreddit: str
    kept: list[dict]
    dropped: list[dict]
    observations: str
    path: str
    input_tokens: int
    output_tokens: int


def _examples(con: duckdb.DuckDBPyConnection, subreddit: str, label: str, window: tuple[str, str], n: int = 8) -> str:
    from modcast.stats import _iso_to_epoch

    rows = con.execute(
        """
        SELECT title, substr(selftext, 1, 300) FROM posts
        WHERE subreddit = ? AND label = ? AND text_available
          AND created_utc >= ? AND created_utc < ?
        ORDER BY hash(id || ?) LIMIT ?
        """,
        [subreddit, label, _iso_to_epoch(window[0]),
         _iso_to_epoch(window[1], exclusive_end=True), str(RANDOM_SEED), n],
    ).fetchall()
    return "\n".join(f"- {r[0]!r}: {r[1]!r}" for r in rows)


def induce(
    con: duckdb.DuckDBPyConnection,
    subreddit: str,
    run_id: str,
    model: str | None = None,
    effort: str | None = None,
    max_turns: int = 20,
) -> InductionResult:
    window = index_window(subreddit)
    base = S.removal_rate(con, subreddit, window=window)
    prompt = "\n\n".join([
        f"Subreddit: r/{subreddit}. Base removal rate: {base['rate']:.1%} over {base['n']} decided posts.",
        "## Published rules\n" + rules_digest(subreddit),
        "## Sample REMOVED posts\n" + _examples(con, subreddit, "removed_mod", window),
        "## Sample SURVIVED posts\n" + _examples(con, subreddit, "survived", window),
        "Begin testing hypotheses.",
    ])
    session = LLMSession(
        run_id=run_id, name=f"induce-{subreddit}", system=SYSTEM, tools=TOOLS,
        effort=effort, **({"model": model} if model else {}),
    )
    response = session.step(prompt)
    entries: list[dict] = []
    observations = ""
    for _ in range(max_turns):
        calls = tool_uses(response)
        if not calls:
            response = session.step("Continue testing, or call finalize_rulebook.")
            continue
        results, done = [], False
        for call in calls:
            if call.name == "finalize_rulebook":
                entries = call.input["entries"]
                observations = call.input["observations"]
                results.append({"type": "tool_result", "tool_use_id": call.id, "content": "received"})
                done = True
            else:
                try:
                    out = S.compare(con, subreddit, where_sql=call.input["where_sql"], window=window)
                    results.append({"type": "tool_result", "tool_use_id": call.id, "content": json.dumps(out)})
                except S.StatsQueryError as e:
                    results.append({"type": "tool_result", "tool_use_id": call.id,
                                    "content": f"where_sql rejected: {e}", "is_error": True})
        session.tool_results(results)
        if done:
            break
        response = session.step()

    # -- code re-verifies every entry before it becomes memory -------------
    kept, dropped = [], []
    for e in entries:
        try:
            cmp = S.compare(con, subreddit, where_sql=e["where_sql"], window=window)
        except S.StatsQueryError as ex:
            dropped.append({**e, "why": f"invalid predicate: {ex}"})
            continue
        t, f = cmp["when_true"], cmp["when_false"]
        effect = t["rate"] - f["rate"]
        if t["n"] < MIN_GROUP_N or f["n"] < MIN_GROUP_N:
            dropped.append({**e, "why": f"group too small (n_true={t['n']}, n_false={f['n']})"})
        elif abs(effect) < MIN_EFFECT:
            dropped.append({**e, "why": f"effect {effect:+.1%} below threshold"})
        else:
            kept.append({**e, "rate_true": t["rate"], "n_true": t["n"],
                         "rate_false": f["rate"], "n_false": f["n"], "effect": effect})

    RULEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    path = RULEBOOK_DIR / f"{subreddit}.md"
    lines = [f"# Rulebook: r/{subreddit}",
             f"Base removal rate {base['rate']:.1%} (n={base['n']}), window {window[0]}..{window[1]}.",
             "Every entry below was verified by code against the labeled corpus.", ""]
    for e in sorted(kept, key=lambda x: -abs(x["effect"])):
        lines.append(
            f"- {e['rule']}\n  evidence: removal {e['rate_true']:.1%} (n={e['n_true']}) when TRUE "
            f"vs {e['rate_false']:.1%} (n={e['n_false']}) when FALSE, effect {e['effect']:+.1%}."
            f"\n  predicate: `{e['where_sql']}`"
            + (f"\n  maps to: {e['maps_to']}" if e.get("maps_to") else "")
        )
    if observations:
        lines += ["", "## Observations", observations]
    path.write_text("\n".join(lines) + "\n")

    return InductionResult(
        subreddit=subreddit, kept=kept, dropped=dropped, observations=observations,
        path=str(path), input_tokens=session.total_input_tokens,
        output_tokens=session.total_output_tokens,
    )
