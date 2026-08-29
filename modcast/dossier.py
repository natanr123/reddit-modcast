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
    removed_contrast: list[dict[str, Any]]   # nearest REMOVED posts (contrast only, not stats)
    neighbor_summary: dict[str, Any]
    feature_values: dict[str, Any]
    rulebook: str                            # induced rulebook markdown ("" if absent)
    published_rules: str

    def to_prompt(self) -> str:
        n = self.neighbor_summary
        neighbor_line = (
            f"- Among the {n['k']} most similar past posts: {n['removed']} removed ({n['rate']:.1%})"
            if n["k"] else "- No sufficiently similar past posts found (unusual post for this subreddit)"
        )
        lines = [
            f"## Candidate post (r/{self.record.subreddit})",
            f"TITLE: {self.record.title}",
            f"BODY:\n{self.record.selftext[:4000]}",
            "",
            "## Computed evidence (code-derived, trustworthy)",
            f"- Subreddit base removal rate ({self.base_rate['n']} posts): "
            f"{self.base_rate['rate']:.1%} [{self.base_rate['ci_low']:.1%}, {self.base_rate['ci_high']:.1%}]",
            neighbor_line,
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
        if self.removed_contrast:
            lines += ["", "## Nearest REMOVED precedents (contrast only — retrieved separately; "
                          "the neighbor statistics above are the unbiased sample)"]
            for nb in self.removed_contrast:
                lines.append(f"- id={nb['id']} sim={nb['score']:.2f} title={nb['title'][:120]!r}")
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
    window: tuple[str, str] | None = None,
) -> Dossier:
    # window matters: regime-shifted subs (config.SUB_INDEX_START) must not
    # quote a base rate contaminated by a dead moderation regime
    base = S.removal_rate(con, record.subreddit, window=window)
    wide = retriever.query(
        f"{record.title}\n\n{record.selftext}", k=max(200, k), before_utc=record.created_utc
    )
    hits = wide[:k]  # the unbiased statistical sample

    def _hydrate(items):
        out = []
        for h in items:
            row = con.execute(
                "SELECT title, substr(selftext, 1, 200) FROM posts WHERE id = ?", [h["id"]]
            ).fetchone()
            out.append({**h, "title": row[0] if row else "", "body": row[1] if row else ""})
        return out

    hydrated = _hydrate(hits[:10])
    # contrast: nearest removed posts from the wider pool, excluding ones already shown
    shown = {h["id"] for h in hits[:10]}
    contrast = _hydrate([h for h in wide if h["label"] == "removed_mod" and h["id"] not in shown][:3])
    return Dossier(
        record=record,
        base_rate=base,
        neighbors=hydrated,
        removed_contrast=contrast,
        neighbor_summary=neighbor_stats(hits),
        feature_values=F.extract(record),
        rulebook=rulebook,
        published_rules=published_rules,
    )
