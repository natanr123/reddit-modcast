"""Citation verifier: no risk factor survives without checkable evidence.

Two evidence types, both machine-checked:
- precedent citations: cited post ids must exist and their fates must
  support the claimed direction;
- statistical claims: a where_sql predicate is RE-EXECUTED against the
  corpus, and the factor survives only if the measured rates support the
  direction (min group size enforced).
A factor whose evidence does not hold is rejected (the agent gets one
repair round, then the factor is dropped from the report).
"""
from __future__ import annotations

from typing import Any

import duckdb

from modcast import stats as S

MIN_STAT_GROUP = 30


def _verify_stat(
    con: duckdb.DuckDBPyConnection, f: dict, subreddit: str, window: tuple[str, str] | None
) -> dict | None:
    """Re-run the factor's statistical claim; return the measured stat or None."""
    try:
        cmp = S.compare(con, subreddit, where_sql=f["stat_where_sql"], window=window)
    except S.StatsQueryError:
        return None
    t, fa = cmp["when_true"], cmp["when_false"]
    if t["n"] < MIN_STAT_GROUP or fa["n"] < MIN_STAT_GROUP or t["rate"] is None or fa["rate"] is None:
        return None
    direction_ok = (t["rate"] > fa["rate"]) if f.get("direction") == "increases" else (t["rate"] < fa["rate"])
    if not direction_ok:
        return None
    return {"rate_true": t["rate"], "n_true": t["n"], "rate_false": fa["rate"], "n_false": fa["n"]}


def verify_factors(
    con: duckdb.DuckDBPyConnection,
    factors: list[dict[str, Any]],
    subreddit: str | None = None,
    window: tuple[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split factors into (verified, rejected) with rejection reasons attached."""
    verified: list[dict] = []
    rejected: list[dict] = []
    for f in factors:
        ids = list(dict.fromkeys(f.get("evidence_post_ids") or []))
        if not ids and f.get("stat_where_sql") and subreddit:
            stat = _verify_stat(con, f, subreddit, window)
            if stat is not None:
                verified.append({**f, "stat": stat})
            else:
                rejected.append({**f, "rejection": "statistical claim failed re-verification "
                                                   "(invalid predicate, small groups, or rates contradict direction)"})
            continue
        if not ids:
            rejected.append({**f, "rejection": "no evidence: cite precedent ids or provide stat_where_sql"})
            continue
        rows = con.execute(
            f"SELECT id, label FROM posts WHERE id IN ({','.join('?' * len(ids))})",
            ids,
        ).fetchall()
        found = {r[0]: r[1] for r in rows}
        missing = [i for i in ids if i not in found]
        if missing:
            rejected.append({**f, "rejection": f"cited ids not in corpus: {missing}"})
            continue
        labels = set(found.values())
        direction = f.get("direction", "increases")
        if direction == "increases" and "removed_mod" not in labels:
            rejected.append({**f, "rejection": "claims increased risk but cites no removed post"})
            continue
        if direction == "decreases" and "survived" not in labels:
            rejected.append({**f, "rejection": "claims decreased risk but cites no surviving post"})
            continue
        verified.append({**f, "evidence_labels": found})
    return verified, rejected
