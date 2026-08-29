"""Deterministic at-post-time features.

Only fields in config.AT_POST_TIME_FIELDS are touched; label inputs and
second-pass fields never appear here. All features are floats, never NaN.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

from modcast.schema import PostRecord

FEATURE_NAMES = [
    "title_len",
    "title_word_count",
    "title_caps_ratio",
    "title_has_question",
    "title_bracket_tag",
    "body_len",
    "body_word_count",
    "body_paragraph_count",
    "body_num_urls",
    "body_upper_ratio",
    "body_first_person_ratio",
    "has_tldr",
    "has_link_flair",
    "has_author_flair",
    "is_self",
    "over_18",
    "spoiler",
    "created_hour_utc",
    "created_weekday",
    "title_body_ratio",
]

_BRACKET_TAG = re.compile(r"^\s*\[?(AITA|WIBTA|UPDATE)", re.IGNORECASE)
_URL = re.compile(r"https?://")
_TLDR = re.compile(r"tl;?\s?dr", re.IGNORECASE)
_WORD = re.compile(r"[a-z']+")
_FIRST_PERSON = {"i", "me", "my"}


def _caps_ratio(text: str) -> float:
    alpha = [c for c in text if c.isalpha()]
    return sum(c.isupper() for c in alpha) / len(alpha) if alpha else 0.0


def extract(record: PostRecord) -> dict[str, float]:
    """Feature dict for one post, keys exactly FEATURE_NAMES in order."""
    title = record.title
    # "[removed]"/"[deleted]" markers are post-removal artifacts, not at-post-time
    # text — treat the body as empty so the marker itself can't leak the label.
    body = record.selftext if record.text_available else ""
    tokens = _WORD.findall(body.lower())
    paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
    dt = datetime.fromtimestamp(record.created_utc, tz=timezone.utc)
    feats = {
        "title_len": float(len(title)),
        "title_word_count": float(len(title.split())),
        "title_caps_ratio": _caps_ratio(title),
        "title_has_question": float("?" in title),
        "title_bracket_tag": float(bool(_BRACKET_TAG.match(title))),
        "body_len": float(len(body)),
        "body_word_count": float(len(body.split())),
        "body_paragraph_count": float(len(paragraphs)),
        "body_num_urls": float(len(_URL.findall(body))),
        "body_upper_ratio": _caps_ratio(body),
        "body_first_person_ratio": (
            sum(t in _FIRST_PERSON for t in tokens) / len(tokens) if tokens else 0.0
        ),
        "has_tldr": float(bool(_TLDR.search(body))),
        "has_link_flair": float(bool((record.link_flair_text or "").strip())),
        "has_author_flair": float(bool((record.author_flair_text or "").strip())),
        "is_self": float(record.is_self),
        "over_18": float(record.over_18),
        "spoiler": float(record.spoiler),
        "created_hour_utc": float(dt.hour),
        "created_weekday": float(dt.weekday()),
        "title_body_ratio": float(len(title)) / len(body) if body else 0.0,
    }
    return {name: feats[name] for name in FEATURE_NAMES}


def featurize_records(records: Iterable[PostRecord]) -> pd.DataFrame:
    """DataFrame of features, columns in FEATURE_NAMES order, no NaNs."""
    df = pd.DataFrame([extract(r) for r in records], columns=FEATURE_NAMES)
    return df.fillna(0.0)
