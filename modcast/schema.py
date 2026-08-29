"""Canonical record shape and labeling.

`normalize()` turns a raw Arctic Shift post JSON into a flat `PostRecord`;
`label()` applies the LabelPolicy. These two functions are the single source
of truth — the fetcher, the duckdb store, and the eval harness all go
through them.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any

from modcast.config import LABEL_POLICY

REMOVED_TEXT_MARKERS = ("", "[removed]", "[deleted]")


class Label(str, Enum):
    SURVIVED = "survived"
    REMOVED_MOD = "removed_mod"        # the positive class
    AUTHOR_DELETED = "author_deleted"  # excluded from the task
    OTHER = "other"                    # unclassifiable; excluded, reported


@dataclass
class PostRecord:
    id: str
    subreddit: str
    created_utc: int
    retrieved_on: int | None
    retrieved_2nd_on: int | None
    title: str
    selftext: str
    is_self: bool
    over_18: bool
    spoiler: bool
    author_flair_text: str | None
    link_flair_text: str | None
    domain: str | None
    url: str | None
    # label inputs (never features!)
    removal_type: str | None          # _meta.removal_type: removed between passes
    removed_by_category: str | None   # reddit's own field at first pass
    was_initially_deleted: bool
    # derived
    label: str
    text_available: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def label(
    removal_type: str | None,
    removed_by_category: str | None,
    was_initially_deleted: bool = False,
) -> Label:
    """Final observable state at the archive's second pass (~36h) decides.

    `was_initially_deleted=True` means the post was removed at the first
    snapshot but is AVAILABLE at the second — i.e. a filter-first automod
    held it and a mod approved it. That is a surviving post; labeling it
    removed was the bug that made filter-first subreddits look like 90%
    removal. (For these restored posts `removal_type` describes the initial
    removal, not the final state.)
    """
    p = LABEL_POLICY
    if was_initially_deleted:
        return Label.SURVIVED
    if removal_type in p.mod_removal_types or removed_by_category in p.mod_removed_by_categories:
        return Label.REMOVED_MOD
    if removal_type in p.author_removal_types or removed_by_category in p.author_removed_by_categories:
        return Label.AUTHOR_DELETED
    if removal_type is None and removed_by_category is None:
        return Label.SURVIVED
    return Label.OTHER


def normalize(raw: dict[str, Any]) -> PostRecord:
    meta = raw.get("_meta") or {}
    removal_type = meta.get("removal_type")
    removed_by_category = raw.get("removed_by_category")
    selftext = raw.get("selftext") or ""
    lab = label(removal_type, removed_by_category, bool(meta.get("was_initially_deleted")))
    return PostRecord(
        id=raw["id"],
        subreddit=raw["subreddit"],
        created_utc=int(raw["created_utc"]),
        retrieved_on=raw.get("retrieved_on"),
        retrieved_2nd_on=meta.get("retrieved_2nd_on"),
        title=raw.get("title") or "",
        selftext=selftext,
        is_self=bool(raw.get("is_self")),
        over_18=bool(raw.get("over_18")),
        spoiler=bool(raw.get("spoiler")),
        author_flair_text=raw.get("author_flair_text"),
        link_flair_text=raw.get("link_flair_text"),
        domain=raw.get("domain"),
        url=raw.get("url"),
        removal_type=removal_type,
        removed_by_category=removed_by_category,
        was_initially_deleted=bool(meta.get("was_initially_deleted")),
        label=lab.value,
        text_available=selftext.strip() not in REMOVED_TEXT_MARKERS,
    )
