"""DuckDB persistence for normalized posts.

One table, `posts`, mirroring schema.PostRecord exactly; all writes go
through schema.normalize so the store never sees un-labeled rows.
"""
from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import duckdb

from modcast.config import DB_PATH
from modcast.schema import PostRecord, normalize

_COLUMNS = [f.name for f in PostRecord.__dataclass_fields__.values()]  # type: ignore[attr-defined]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
    id TEXT PRIMARY KEY,
    subreddit TEXT NOT NULL,
    created_utc BIGINT NOT NULL,
    retrieved_on BIGINT,
    retrieved_2nd_on BIGINT,
    title TEXT NOT NULL,
    selftext TEXT NOT NULL,
    is_self BOOLEAN NOT NULL,
    over_18 BOOLEAN NOT NULL,
    spoiler BOOLEAN NOT NULL,
    author_flair_text TEXT,
    link_flair_text TEXT,
    domain TEXT,
    url TEXT,
    removal_type TEXT,
    removed_by_category TEXT,
    was_initially_deleted BOOLEAN NOT NULL,
    label TEXT NOT NULL,
    text_available BOOLEAN NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_posts_sub_created ON posts (subreddit, created_utc);
"""


class Store:
    """DuckDB-backed post store. Pass db_path=":memory:" or a tmp path for tests."""

    def __init__(self, db_path: str | Path = DB_PATH):
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(db_path))
        self.con.execute(_SCHEMA)

    def ingest_raw(self, raws: Iterable[dict[str, Any]]) -> dict[str, int]:
        """Normalize and upsert raw archive dicts; returns per-label counts + total."""
        rows = [tuple(getattr(normalize(raw), c) for c in _COLUMNS) for raw in raws]
        if rows:
            placeholders = ", ".join("?" for _ in _COLUMNS)
            self.con.executemany(
                f"INSERT OR REPLACE INTO posts VALUES ({placeholders})", rows
            )
        counts = Counter(row[_COLUMNS.index("label")] for row in rows)
        return {"total": len(rows), **counts}

    def ingest_jsonl_gz(self, path: str | Path) -> dict[str, int]:
        """Ingest a gzipped JSONL file of raw posts (one JSON object per line)."""
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return self.ingest_raw(json.loads(line) for line in f if line.strip())

    def counts(self, subreddit: str | None = None) -> dict[tuple[str, bool], int]:
        """Label x text_available counts, optionally for one subreddit."""
        sql = "SELECT label, text_available, count(*) FROM posts"
        params: list[Any] = []
        if subreddit is not None:
            sql += " WHERE subreddit = ?"
            params.append(subreddit)
        sql += " GROUP BY label, text_available"
        return {(lab, avail): n for lab, avail, n in self.con.execute(sql, params).fetchall()}

    def query(self, sql: str, params: list[Any] | None = None) -> duckdb.DuckDBPyConnection:
        """Raw SQL passthrough; chain .fetchall() / .df() on the result."""
        return self.con.execute(sql, params or [])

    def close(self) -> None:
        self.con.close()
