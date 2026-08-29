"""Evidence dossier: the compact, computed context the analyst model reasons over.

Context engineering rule: the model never sees raw corpus dumps — it sees
numbers computed by code (neighbor removal stats, base rates, feature values)
plus short snippets of the nearest precedents it may cite.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import duckdb

from modcast import features as F
from modcast import stats as S
from modcast.config import RETRIEVAL_TOP_K
from modcast.retrieval import TfidfRetriever, neighbor_stats
from modcast.schema import PostRecord


@dataclass
class Dossier:
    record: PostRecord
    base_rate: dict[str, Any]
    neighbors: list[dict[str, Any]]          # hydrated: + title, label, snippet
    neighbor_summary: dict[str, Any]
    feature_values: dict[str, Any]
    rulebook: str                            # induced rulebook markdown ("" if absent)
    published_rules: str

    def to_prompt(self) -> str:
        n = self.neighbor_summary
        lines = [
            f"## Candidate post (r/{self.record.subreddit})",
            f"TITLE: {self.record.title}",
            f"BODY:\n{self.record.selftext[:4000]}",
            "",
            "## Computed evidence (code-derived, trustworthy)",
            f"- Subreddit base removal rate ({self.base_rate['n']} posts): "
            f"{self.base_rate['rate']:.1%} [{self.base_rate['ci_low']:.1%}, {self.base_rate['ci_high']:.1%}]",
            f"- Among the {n['k']} most similar past posts: {n['removed']} removed "
            f"({n['rate']:.1%})",
            "- Deterministic features: "
            + ", ".join(f"{k}={v}" for k, v in sorted(self.feature_values.items())),
            "",
            "## Nearest precedent posts (cite these ids as evidence)",
        ]
        for nb in self.neighbors:
            lines.append(
                f"- id={nb['id']} sim={nb['score']:.2f} label={nb['label']} "
                f"title={nb['title'][:120]!r}"
            )
        if self.rulebook:
            lines += ["", "## Induced rulebook (statistically verified on this subreddit)", self.rulebook]
        lines += ["", "## Published subreddit rules", self.published_rules]
        return "\n".join(lines)


def build(
    con: duckdb.DuckDBPyConnection,
    retriever: TfidfRetriever,
    record: PostRecord,
    rulebook: str = "",
    published_rules: str = "",
    k: int = RETRIEVAL_TOP_K,
) -> Dossier:
    base = S.removal_rate(con, record.subreddit)
    hits = retriever.query(
        f"{record.title}\n\n{record.selftext}", k=k, before_utc=record.created_utc
    )
    hydrated = []
    for h in hits[:10]:
        row = con.execute(
            "SELECT title FROM posts WHERE id = ?", [h["id"]]
        ).fetchone()
        hydrated.append({**h, "title": row[0] if row else ""})
    return Dossier(
        record=record,
        base_rate=base,
        neighbors=hydrated,
        neighbor_summary=neighbor_stats(hits),
        feature_values=F.extract(record),
        rulebook=rulebook,
        published_rules=published_rules,
    )
