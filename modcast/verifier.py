"""Citation verifier: no risk factor survives without checkable evidence.

The analyst must cite precedent post ids for every claimed risk factor.
This module independently checks the citations against the database; a
factor whose evidence does not hold is rejected (the agent gets one chance
to repair, then the factor is dropped from the report).
"""
from __future__ import annotations

from typing import Any

import duckdb


def verify_factors(
    con: duckdb.DuckDBPyConnection, factors: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split factors into (verified, rejected) with rejection reasons attached."""
    verified: list[dict] = []
    rejected: list[dict] = []
    for f in factors:
        ids = list(dict.fromkeys(f.get("evidence_post_ids") or []))
        if not ids:
            rejected.append({**f, "rejection": "no evidence ids cited"})
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
