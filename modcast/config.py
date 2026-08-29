"""Central configuration: subreddits, date splits, paths, label policy.

Every module reads from here; nothing hardcodes dates or subreddit names.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Load PROJECT_ROOT/.env into os.environ (existing env vars win)."""
    path = PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "modcast.duckdb"

# Generated artifacts (eval results, figures, trajectories, rulebooks, reports)
# go under MODCAST_OUT_DIR while experimenting (e.g. tmp/generated, gitignored);
# unset it — the judge default — and they land in the committed repo folders.
_out = os.environ.get("MODCAST_OUT_DIR")
OUT_ROOT = (PROJECT_ROOT / _out).resolve() if _out else PROJECT_ROOT
RESULTS_DIR = OUT_ROOT / "results"
RULEBOOK_DIR = OUT_ROOT / "rulebooks"

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

# Regime shifts: r/AmItheAsshole ran filter-first moderation (automod removed
# ~95% of new posts at creation; mods approved a subset later) until late June
# 2026. Training on that regime would model a policy that no longer exists at
# test time, so its index window starts where the current regime does.
SUB_INDEX_START: dict[str, str] = {"AmItheAsshole": "2026-07-01"}


def index_window(subreddit: str) -> tuple[str, str]:
    return (SUB_INDEX_START.get(subreddit, INDEX_START), INDEX_END)

# LLM backend selection. `.env`: LLM_MODEL=<backend> <model> <effort>
#   codex-cli gpt-5.6-sol medium   -> flat-rate codex CLI (free test iterations)
#   anthropic claude-opus-5 high   -> metered API (judge default when unset)
_llm_parts = os.environ.get("LLM_MODEL", "").split()
LLM_BACKEND = _llm_parts[0] if _llm_parts else "anthropic"
LLM_MODEL_NAME = (
    _llm_parts[1] if len(_llm_parts) > 1 else os.environ.get("MODCAST_MODEL", "claude-opus-5")
)
LLM_EFFORT = _llm_parts[2] if len(_llm_parts) > 2 else "high"

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
