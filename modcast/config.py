"""Central configuration: subreddits, date splits, paths, label policy.

Every module reads from here; nothing hardcodes dates or subreddit names.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "modcast.duckdb"
RESULTS_DIR = PROJECT_ROOT / "results"
RULEBOOK_DIR = PROJECT_ROOT / "rulebooks"

ARCTIC_SHIFT_BASE = "https://arctic-shift.photon-reddit.com/api"
USER_AGENT = "reddit-modcast (micro1 hackathon research; contact: repo issues)"

# Subreddits chosen after the 2026-08-20 probe (see results/go_no_go.md):
# balance of removal rate (20-75%), text availability (54-93%), and
# distinct moderation styles (topic rules / format rules / heavy automod).
SUBREDDITS = [
    "AmItheAsshole",
    "legaladvice",
    "personalfinance",
    "unpopularopinion",
]

# Temporal split. The index/train window feeds retrieval, rule induction and
# baseline fitting; the test window is strictly later (no leakage). Both end
# well before "now" minus 36h so every post has its second-pass labels.
INDEX_START = "2026-01-01"
INDEX_END = "2026-07-31"  # inclusive
TEST_START = "2026-08-01"
TEST_END = "2026-08-25"  # inclusive

# Evaluation
EVAL_POSTS_PER_SUB = 250          # sampled from the test window per subreddit
LLM_EVAL_POSTS_PER_SUB = 60       # seeded subsample of the above for LLM predictors
RANDOM_SEED = 20260831            # used everywhere sampling happens
RETRIEVAL_TOP_K = 25


@dataclass(frozen=True)
class LabelPolicy:
    """How raw archive records map to the prediction target.

    A post is REMOVED_MOD when moderation (human mod, automod, or reddit
    admin/spam) took it down, either before the archive's first snapshot
    (visible as `removed_by_category` on the first pass) or between the
    first and second pass (visible as `_meta.removal_type`).

    Author self-deletes are NOT a moderation outcome and are excluded from
    the task entirely. Posts removed before first snapshot usually have no
    body text (text_available=False); they keep their label but are excluded
    from text-based evaluation and reported separately for honesty.
    """

    mod_removal_types: tuple[str, ...] = ("moderator", "reddit")
    mod_removed_by_categories: tuple[str, ...] = (
        "moderator",
        "automod_filtered",
        "reddit",
        "content_takedown",
        "copyright_takedown",
    )
    author_removal_types: tuple[str, ...] = ("deleted",)
    author_removed_by_categories: tuple[str, ...] = ("deleted", "author")


LABEL_POLICY = LabelPolicy()

# The model may only use information visible at post time.  Second-pass
# fields (score, num_comments, upvote_ratio, locked, ...) are forbidden as
# features; this list is what feature extraction is allowed to touch.
AT_POST_TIME_FIELDS = [
    "title",
    "selftext",
    "subreddit",
    "author_flair_text",
    "link_flair_text",
    "is_self",
    "over_18",
    "spoiler",
    "url",
    "domain",
    "created_utc",
]
