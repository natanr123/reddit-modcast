"""Verified aggregate stats over the duckdb `posts` table.

Agent-facing: the LLM calls these instead of eyeballing rows, and may pass
an SQL boolean fragment (`where_sql`) that is strictly validated before
execution. Rates are computed ONLY over decided labels (survived,
removed_mod); author-deletes and unknowns never enter a denominator.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import duckdb

DECIDED_LABELS = ("survived", "removed_mod")

ALLOWED_COLUMNS = frozenset({
    "title", "selftext", "is_self", "over_18", "link_flair_text",
    "author_flair_text", "created_utc", "text_available",
})
ALLOWED_FUNCTIONS = frozenset({
    "len", "length", "lower", "upper", "regexp_matches", "strftime", "epoch",
})
ALLOWED_KEYWORDS = frozenset({
    "and", "or", "not", "like", "ilike", "is", "null", "true", "false",
})

_TOKEN_RE = re.compile(
    r"""\s+
      | '(?:[^']|'')*'                # string literal
      | \d+(?:\.\d+)?                 # number
      | [A-Za-z_][A-Za-z0-9_]*        # identifier / keyword
      | <> | != | <= | >=
      | [<>=(),?*/%+-]
    """,
    re.VERBOSE,
)


class StatsQueryError(ValueError):
    """Raised when a where_sql fragment fails validation or execution."""


def validate_where_sql(where_sql: str) -> str:
    """Return the fragment if safe, else raise StatsQueryError.

    Allows only the at-post-time columns, AND/OR/NOT/IS NULL, comparisons,
    LIKE/ILIKE, basic arithmetic, string/number literals, `?` placeholders,
    and the functions len/length/lower/upper/regexp_matches/strftime/epoch.
    """
    if not where_sql or not where_sql.strip():
        raise StatsQueryError("where_sql is empty; pass a boolean SQL fragment.")
    if ";" in where_sql:
        raise StatsQueryError("semicolons are not allowed in where_sql.")
    if "--" in where_sql or "/*" in where_sql:
        raise StatsQueryError("SQL comments are not allowed in where_sql.")
    allowed = ALLOWED_COLUMNS | ALLOWED_FUNCTIONS | ALLOWED_KEYWORDS
    pos = 0
    while pos < len(where_sql):
        m = _TOKEN_RE.match(where_sql, pos)
        if m is None:
            raise StatsQueryError(
                f"illegal character {where_sql[pos]!r} at position {pos} in where_sql."
            )
        tok = m.group(0)
        if tok[0].isalpha() or tok[0] == "_":
            if tok.lower() not in allowed:
                raise StatsQueryError(
                    f"identifier {tok!r} is not allowed. Allowed columns: "
                    f"{sorted(ALLOWED_COLUMNS)}; functions: {sorted(ALLOWED_FUNCTIONS)}; "
                    f"keywords: {sorted(ALLOWED_KEYWORDS)}. No subqueries."
                )
        pos = m.end()
    return where_sql


def _iso_to_epoch(iso: str, *, exclusive_end: bool = False) -> int:
    """ISO date/datetime -> epoch seconds (UTC). Date-only ends are inclusive."""
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if exclusive_end and len(iso) == 10:
        dt += timedelta(days=1)
    return int(dt.timestamp())


def _wilson(removed: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = removed / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def removal_rate(
    con: duckdb.DuckDBPyConnection,
    subreddit: str,
    where_sql: str | None = None,
    params: tuple = (),
    window: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Removal rate with Wilson 95% CI over decided posts.

    `window` is (start_iso, end_iso) on created_utc, end date inclusive.
    Returns {n, removed, rate, ci_low, ci_high}; rate/CI are None when n=0.
    """
    sql = (
        "SELECT count(*), count(*) FILTER (WHERE label = 'removed_mod') "
        "FROM posts WHERE subreddit = ? AND label IN (?, ?)"
    )
    bind: list[Any] = [subreddit, *DECIDED_LABELS]
    if window is not None:
        sql += " AND created_utc >= ? AND created_utc < ?"
        bind += [_iso_to_epoch(window[0]), _iso_to_epoch(window[1], exclusive_end=True)]
    if where_sql is not None:
        sql += f" AND ({validate_where_sql(where_sql)})"
        bind += list(params)
    try:
        n, removed = con.execute(sql, bind).fetchone()
    except duckdb.Error as e:
        raise StatsQueryError(f"where_sql failed to execute: {e}") from e
    if n == 0:
        return {"n": 0, "removed": 0, "rate": None, "ci_low": None, "ci_high": None}
    lo, hi = _wilson(removed, n)
    return {
        "n": n,
        "removed": removed,
        "rate": round(removed / n, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
    }


def compare(
    con: duckdb.DuckDBPyConnection,
    subreddit: str,
    where_sql: str,
    params: tuple = (),
    window: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Removal rate when the condition holds vs. when it does not.

    `lift` is rate_true / rate_false (None if either side is undefined or
    the false-side rate is 0). Rows where the condition is NULL fall in
    neither bucket.
    """
    cond = validate_where_sql(where_sql)
    when_true = removal_rate(con, subreddit, cond, params, window)
    when_false = removal_rate(con, subreddit, f"NOT ({cond})", params, window)
    rt, rf = when_true["rate"], when_false["rate"]
    lift = round(rt / rf, 4) if rt is not None and rf else None
    return {
        "when_true": when_true,
        "when_false": when_false,
        "lift": lift,
        "n_true": when_true["n"],
        "n_false": when_false["n"],
    }
