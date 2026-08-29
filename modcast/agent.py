"""The ModCast forecast agent.

A manual tool loop (transparent, fully logged): the analyst model receives a
code-computed evidence dossier, may interrogate the corpus through verified
tools (statistical queries, similarity search, precedent reading), and must
deliver its forecast through a strict-schema tool. Every cited risk factor is
then independently checked by the citation verifier; the model gets one
repair round before unverified factors are dropped.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import duckdb

from modcast import stats as S
from modcast import verifier
from modcast.dossier import Dossier, build as build_dossier
from modcast.llm import LLMSession, tool_uses
from modcast.retrieval import TfidfRetriever, neighbor_stats
from modcast.schema import PostRecord

MAX_TURNS = 12

SYSTEM_TEMPLATE = """You are ModCast, a moderation-risk forecaster for Reddit posts.
Your job: estimate the probability that the candidate post will be REMOVED BY
MODERATION (human mods, automod, or admins — author self-deletes don't count)
within 36 hours of posting, in the given subreddit.

Your probability is scored with the Brier score against real outcomes, so be
calibrated, not dramatic: anchor on the subreddit base rate, then adjust for
evidence. Only computed evidence moves you far from the anchor.

Trust hierarchy:
1. Numbers computed by tools (removal rates, comparisons, neighbor stats) are
   ground truth from the real labeled corpus.
2. The induced rulebook entries were statistically verified — trust them.
3. Published subreddit rules describe intent; automod behavior may differ.
4. Your own priors about Reddit are the weakest signal.

{method}

where_sql mini-language (validated; errors explain themselves — fix and retry):
columns: title, selftext, is_self, over_18, link_flair_text, author_flair_text,
created_utc, text_available. functions: length, lower, upper, regexp_matches,
strftime, epoch. operators: AND OR NOT LIKE ILIKE = != < > <= >= IS NULL.
Example: NOT regexp_matches(lower(title), 'aita|wibta')"""

METHOD_V1 = """Method:
- Read the dossier. Identify which rulebook entries and published rules the
  post might trip.
- Test hypotheses about THIS post's traits with compare_removal_rate (e.g. if
  the post has no flair, compare removal for flaired vs unflaired posts).
- Read 2-4 nearest precedents (read_post) and note their fates.
- Then call submit_forecast. Every risk factor must cite precedent post ids
  that actually support it — citations are machine-verified and unverifiable
  factors are discarded, so cite only what you have seen in tool results."""

METHOD_V2 = """Method — investigation is mandatory, the dossier is only the backdrop:
- The base rate, rulebook and neighbor stats describe the AVERAGE post here.
  Your job is to find what distinguishes THIS post from that average. A
  forecast justified only by subreddit-level facts (rules that apply to most
  posts equally) is a failed forecast stuck at the base rate.
- ALWAYS, before submitting: read 2-4 of the most similar precedents
  (read_post) and note their fates; and test at least 2 hypotheses SPECIFIC
  to this post's distinctive traits with compare_removal_rate.
- Rulebook entries move the anchor, not the verdict: weight post-specific
  evidence (precedent fates, targeted comparisons) above them.
- Then call submit_forecast. Every risk factor must cite precedent post ids
  that actually support it — citations are machine-verified and unverifiable
  factors are discarded, so cite only what you have seen in tool results."""

# Prompt variant is an ablation axis (see changelog): v1 = anchor-first,
# v2 = investigation-first. Select via MODCAST_PROMPT; cache keys hash SYSTEM,
# so variants never mix cached forecasts.
PROMPT_VARIANT = os.environ.get("MODCAST_PROMPT", "v1")  # v1 measured best (see changelog)
SYSTEM = SYSTEM_TEMPLATE.format(method=METHOD_V1 if PROMPT_VARIANT == "v1" else METHOD_V2)

TOOLS: list[dict] = [
    {
        "name": "query_removal_rate",
        "description": "Removal rate among decided posts (survived vs removed_mod) in the subreddit's index window, optionally restricted by a where_sql predicate. Returns n, removed, rate and a 95% Wilson interval.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "where_sql": {"type": ["string", "null"], "description": "Optional boolean predicate in the where_sql mini-language; null for the overall base rate."},
                "reason": {"type": "string", "description": "One line: which hypothesis this tests."},
            },
            "required": ["where_sql", "reason"],
            "additionalProperties": False,
        },
    },
    {
        "name": "compare_removal_rate",
        "description": "Compare removal rate where the predicate is TRUE vs FALSE. Returns both groups with counts and the lift. The strongest evidence tool — use it to test whether a trait of the candidate post matters in this subreddit.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "where_sql": {"type": "string", "description": "Boolean predicate in the where_sql mini-language."},
                "reason": {"type": "string", "description": "One line: which hypothesis this tests."},
            },
            "required": ["where_sql", "reason"],
            "additionalProperties": False,
        },
    },
    {
        "name": "find_similar_posts",
        "description": "Similarity search over the subreddit's past posts (index window only). Returns ids, similarity, labels and titles.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "k": {"type": "integer", "description": "How many neighbors, 1-50."},
            },
            "required": ["text", "k"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_post",
        "description": "Read one archived post by id: title, body excerpt, label, flair.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"post_id": {"type": "string"}},
            "required": ["post_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "submit_forecast",
        "description": "Deliver the final forecast. Call exactly once, after your investigation.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "p_removed": {"type": "number", "description": "Probability in [0, 1]."},
                "risk_factors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "factor": {"type": "string", "description": "The trait of THIS post and why it matters here."},
                            "direction": {"type": "string", "enum": ["increases", "decreases"]},
                            "evidence_post_ids": {"type": "array", "items": {"type": "string"}, "description": "Precedent post ids you have seen in tool results. May be empty ONLY when stat_where_sql is given."},
                            "rule_ref": {"type": ["string", "null"], "description": "Rulebook entry or published rule this maps to, if any."},
                            "stat_where_sql": {"type": ["string", "null"], "description": "Alternative evidence: a where_sql predicate capturing this post's trait. The verifier RE-RUNS it on the corpus and keeps the factor only if the measured rates support the direction. Use for statistically-backed factors no read precedent demonstrates (e.g. the aggregate short-body effect when every inspected precedent survived)."},
                        },
                        "required": ["factor", "direction", "evidence_post_ids", "rule_ref", "stat_where_sql"],
                        "additionalProperties": False,
                    },
                },
                "suggested_fixes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Concrete edits that would lower removal risk while keeping the post honest. Empty if the post is fine — or if it violates rules outright and no cosmetic edit should save it (say so in reasoning_summary instead).",
                },
                "reasoning_summary": {"type": "string"},
            },
            "required": ["p_removed", "risk_factors", "suggested_fixes", "reasoning_summary"],
            "additionalProperties": False,
        },
    },
]


@dataclass
class AgentContext:
    con: duckdb.DuckDBPyConnection
    retriever: TfidfRetriever
    subreddit: str
    window: tuple[str, str] | None = None  # restrict stats to the index window


@dataclass
class Forecast:
    p_removed: float
    risk_factors: list[dict]
    rejected_factors: list[dict]
    suggested_fixes: list[str]
    reasoning_summary: str
    turns: int
    input_tokens: int
    output_tokens: int
    raw_submission: dict = field(default_factory=dict)
    neighbors: list[dict] = field(default_factory=list)  # dossier's hydrated similar posts
    removed_contrast: list[dict] = field(default_factory=list)
    neighbor_summary: dict = field(default_factory=dict)     # {k, removed, rate} over all retrieved


def _execute_tool(ctx: AgentContext, name: str, args: dict) -> tuple[str, bool]:
    """Returns (content, is_error)."""
    try:
        if name == "query_removal_rate":
            out = S.removal_rate(ctx.con, ctx.subreddit, where_sql=args.get("where_sql"), window=ctx.window)
            return json.dumps(out), False
        if name == "compare_removal_rate":
            out = S.compare(ctx.con, ctx.subreddit, where_sql=args["where_sql"], window=ctx.window)
            return json.dumps(out), False
        if name == "find_similar_posts":
            hits = ctx.retriever.query(args["text"], k=max(1, min(50, int(args["k"]))))
            for h in hits:
                row = ctx.con.execute("SELECT title FROM posts WHERE id = ?", [h["id"]]).fetchone()
                h["title"] = (row[0] if row else "")[:140]
            return json.dumps({"hits": hits, "summary": neighbor_stats(hits)}), False
        if name == "read_post":
            row = ctx.con.execute(
                "SELECT title, selftext, label, link_flair_text FROM posts WHERE id = ?",
                [args["post_id"]],
            ).fetchone()
            if not row:
                return f"no post with id {args['post_id']} in corpus", True
            return json.dumps({
                "title": row[0], "body_excerpt": (row[1] or "")[:2500],
                "label": row[2], "link_flair_text": row[3],
            }), False
        return f"unknown tool {name}", True
    except S.StatsQueryError as e:
        return f"where_sql rejected: {e}", True
    except Exception as e:  # tool bugs must not kill the run; surface to model
        return f"tool error: {type(e).__name__}: {e}", True


def forecast(
    ctx: AgentContext,
    record: PostRecord,
    run_id: str,
    rulebook: str = "",
    published_rules: str = "",
    model: str | None = None,
    effort: str | None = None,
) -> Forecast:
    dossier: Dossier = build_dossier(
        ctx.con, ctx.retriever, record, rulebook=rulebook,
        published_rules=published_rules, window=ctx.window,
    )
    session = LLMSession(
        run_id=run_id,
        name=f"forecast-{record.id}",
        system=SYSTEM,
        tools=TOOLS,
        effort=effort,
        **({"model": model} if model else {}),
    )
    response = session.step(dossier.to_prompt())
    submission: dict | None = None
    repair_rounds = 0

    for turn in range(MAX_TURNS):
        calls = tool_uses(response)
        if not calls:
            if response.stop_reason == "refusal":
                break
            # nudge toward the required contract
            response = session.step("Continue your investigation, or call submit_forecast.")
            continue
        results = []
        for call in calls:
            if call.name == "submit_forecast":
                submission = call.input
                verified, rejected = verifier.verify_factors(ctx.con, submission.get("risk_factors", []), subreddit=ctx.subreddit, window=ctx.window)
                if rejected and repair_rounds == 0:
                    repair_rounds += 1
                    submission = None
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": "Citation verification FAILED for some factors:\n"
                        + json.dumps(rejected, default=str)
                        + "\nResubmit with corrected evidence ids (or drop those factors).",
                        "is_error": True,
                    })
                else:
                    submission = {**submission, "_verified": verified, "_rejected": rejected}
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": "forecast accepted",
                    })
            else:
                content, is_err = _execute_tool(ctx, call.name, call.input)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": content,
                    **({"is_error": True} if is_err else {}),
                })
        session.tool_results(results)
        if submission is not None:
            break
        response = session.step()

    if submission is None:  # ran out of turns or refusal: fall back to neighbor evidence
        base = dossier.neighbor_summary["rate"] if dossier.neighbor_summary["k"] else dossier.base_rate["rate"]
        submission = {
            "p_removed": base, "risk_factors": [], "suggested_fixes": [],
            "reasoning_summary": "fallback: agent did not submit; using neighbor removal rate",
            "_verified": [], "_rejected": [],
        }
    return Forecast(
        p_removed=float(min(1.0, max(0.0, submission["p_removed"]))),
        risk_factors=submission.get("_verified", []),
        rejected_factors=submission.get("_rejected", []),
        suggested_fixes=submission.get("suggested_fixes", []),
        reasoning_summary=submission.get("reasoning_summary", ""),
        turns=turn + 1,
        input_tokens=session.total_input_tokens,
        output_tokens=session.total_output_tokens,
        raw_submission={k: v for k, v in submission.items() if not k.startswith("_")},
        neighbors=dossier.neighbors,
        removed_contrast=dossier.removed_contrast,
        neighbor_summary=dossier.neighbor_summary,
    )
