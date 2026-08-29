"""Arctic Shift acquisition: paginated search, gzip JSONL cache, id lookup.

All timestamps are epoch seconds (the API also accepts YYYY-MM-DD). The
API's `after` bound is exclusive, so a window [after, before) paginates by
advancing `after` to the last created_utc seen, until an empty page.
"""
from __future__ import annotations

import gzip
import json
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

from modcast.config import ARCTIC_SHIFT_BASE, RAW_DIR, USER_AGENT

PAGE_SLEEP = 1.0       # politeness delay between requests
MAX_RETRIES = 5        # for 5xx responses
IDS_PER_REQUEST = 500  # API cap for /posts/ids


class ArcticShiftClient:
    """httpx client for the Arctic Shift REST API with a resumable local cache."""

    def __init__(
        self,
        base_url: str = ARCTIC_SHIFT_BASE,
        raw_dir: Path = RAW_DIR,
        page_sleep: float = PAGE_SLEEP,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.raw_dir = raw_dir
        self.page_sleep = page_sleep
        self._client = httpx.Client(
            headers={"User-Agent": USER_AGENT}, timeout=60, transport=transport
        )

    def close(self) -> None:
        self._client.close()

    # -- HTTP ----------------------------------------------------------------

    def _get(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        """GET with rate-limit (429) waits and 5xx backoff retries."""
        attempts = 0
        while True:
            resp = self._client.get(self.base_url + path, params=params)
            if resp.status_code == 429:
                time.sleep(float(resp.headers.get("X-RateLimit-Reset", "10")) + 1)
                continue
            if resp.status_code >= 500 and attempts < MAX_RETRIES:
                attempts += 1
                time.sleep(2**attempts)
                continue
            resp.raise_for_status()
            return resp.json()

    def search_page(self, subreddit: str, after: int, before: int) -> list[dict[str, Any]]:
        """One ascending page of posts; `after` is exclusive, `before` exclusive.

        limit=auto can 422 on very small residual windows; fall back to a
        fixed limit, and treat a persistent 422 on a tiny window as empty.
        """
        params = {
            "subreddit": subreddit,
            "after": str(after),
            "before": str(before),
            "limit": "auto",
            "sort": "asc",
        }
        try:
            return self._get("/posts/search", params)["data"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 422:
                raise
        try:
            return self._get("/posts/search", {**params, "limit": "100"})["data"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 422 and before - after < 600:
                return []  # sub-10-minute residual window the API refuses; nothing left
            raise

    def iter_window(self, subreddit: str, after: int, before: int) -> Iterator[dict[str, Any]]:
        """Every post in [after, before), paginating until an empty page."""
        cursor = after
        while True:
            page = self.search_page(subreddit, cursor, before)
            if not page:
                return
            yield from page
            cursor = int(page[-1]["created_utc"])
            time.sleep(self.page_sleep)

    def fetch_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Look up posts by base36 id via /posts/ids, 500 per request."""
        out: list[dict[str, Any]] = []
        for i in range(0, len(ids), IDS_PER_REQUEST):
            chunk = ids[i : i + IDS_PER_REQUEST]
            out.extend(self._get("/posts/ids", {"ids": ",".join(chunk)})["data"])
            if i + IDS_PER_REQUEST < len(ids):
                time.sleep(self.page_sleep)
        return out

    # -- cache ---------------------------------------------------------------

    def _cache_path(self, subreddit: str) -> Path:
        return self.raw_dir / f"{subreddit}.jsonl.gz"

    def _manifest_path(self, subreddit: str) -> Path:
        return self.raw_dir / f"{subreddit}.manifest.json"

    def _completed_windows(self, subreddit: str) -> list[list[int]]:
        path = self._manifest_path(subreddit)
        if not path.exists():
            return []
        return json.loads(path.read_text())["windows"]

    def window_done(self, subreddit: str, after: int, before: int) -> bool:
        """True if [after, before) lies inside a completed window."""
        return any(a <= after and before <= b for a, b in self._completed_windows(subreddit))

    def fetch_window(self, subreddit: str, after: int, before: int) -> int:
        """Fetch one window into the cache; no-op if the manifest covers it.

        The manifest is written only after the window completes, so an
        interrupted run re-fetches it (iter_cached dedupes any repeats).
        Returns the number of posts appended.
        """
        if self.window_done(subreddit, after, before):
            return 0
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        with gzip.open(self._cache_path(subreddit), "at", encoding="utf-8") as f:
            for post in self.iter_window(subreddit, after, before):
                f.write(json.dumps(post) + "\n")
                n += 1
        windows = self._completed_windows(subreddit)
        windows.append([after, before])
        self._manifest_path(subreddit).write_text(
            json.dumps({"windows": sorted(windows)}, indent=1)
        )
        return n


def iter_cached(subreddit: str, raw_dir: Path = RAW_DIR) -> Iterator[dict[str, Any]]:
    """Yield raw post dicts from the cache, deduped by id (first wins)."""
    path = raw_dir / f"{subreddit}.jsonl.gz"
    if not path.exists():
        return
    seen: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            post = json.loads(line)
            if post["id"] not in seen:
                seen.add(post["id"])
                yield post


# -- URL mode ----------------------------------------------------------------

_POST_ID_RE = re.compile(r"(?:comments|redd\.it)/([a-z0-9]{4,10})", re.I)


def post_id_from_url(url: str) -> str:
    m = _POST_ID_RE.search(url)
    if not m:
        raise ValueError(f"cannot find a post id in {url!r}")
    return m.group(1).lower()


def post_from_url(url: str) -> dict[str, Any]:
    """Fetch one post's raw JSON given any reddit post URL."""
    return post_by_id(post_id_from_url(url))


def post_by_id(pid: str) -> dict[str, Any]:
    """Fetch one post's raw JSON by base36 id.

    reddit's public .json endpoint first (freshest); when the post is already
    removed/deleted there (or the request fails), fall back to the Arctic
    Shift archive, which preserves the pre-removal text.
    """
    raw: dict[str, Any] | None = None
    try:
        r = httpx.get(
            f"https://www.reddit.com/comments/{pid}.json",
            headers={"User-Agent": USER_AGENT},
            timeout=30, follow_redirects=True,
        )
        r.raise_for_status()
        raw = r.json()[0]["data"]["children"][0]["data"]
    except Exception:
        raw = None
    if raw is None or (raw.get("selftext") or "") in ("[removed]", "[deleted]"):
        client = ArcticShiftClient()
        try:
            archived = client.fetch_by_ids([pid])
        finally:
            client.close()
        if archived:
            raw = archived[0]
    if raw is None:
        raise ValueError(f"post {pid} not found on reddit or in the archive")
    return raw
