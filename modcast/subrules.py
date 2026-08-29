"""Fetch and cache a subreddit's published rules (public JSON endpoint).

The induction step grounds hypotheses in the rules mods actually publish;
the preflight report cites them. Cached under data/subrules/ so eval runs
and judges never depend on reddit.com being reachable.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx

from modcast.config import DATA_DIR, USER_AGENT

SUBRULES_DIR = DATA_DIR / "subrules"


def get_rules(subreddit: str, refresh: bool = False) -> list[dict]:
    """Return [{short_name, description, ...}] for a subreddit; [] on failure."""
    cache = SUBRULES_DIR / f"{subreddit}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    try:
        r = httpx.get(
            f"https://www.reddit.com/r/{subreddit}/about/rules.json",
            headers={"User-Agent": USER_AGENT},
            timeout=30,
            follow_redirects=True,
        )
        r.raise_for_status()
        rules = r.json().get("rules", [])
    except Exception:
        rules = []
    if rules:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rules, indent=1))
    return rules


def rules_digest(subreddit: str) -> str:
    """Compact plain-text rendering for prompts."""
    rules = get_rules(subreddit)
    if not rules:
        return "(no published rules retrieved)"
    lines = []
    for i, r in enumerate(rules, 1):
        desc = (r.get("description") or "").strip().replace("\n", " ")
        if len(desc) > 300:
            desc = desc[:300] + "…"
        lines.append(f"{i}. {r.get('short_name', '')} — {desc}")
    return "\n".join(lines)
