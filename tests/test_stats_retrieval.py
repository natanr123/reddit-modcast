"""Fixture-driven tests for modcast.stats and modcast.retrieval."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from modcast import config, schema
from modcast.retrieval import TfidfRetriever, neighbor_stats
from modcast.stats import StatsQueryError, compare, removal_rate, validate_where_sql

FIXTURE = Path(__file__).parent / "fixtures" / "sample_aita.json"


@pytest.fixture(scope="module")
def records() -> list[schema.PostRecord]:
    return [schema.normalize(raw) for raw in json.loads(FIXTURE.read_text())]


@pytest.fixture(scope="module")
def con(records) -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    df = pd.DataFrame([r.to_dict() for r in records])
    c.register("df", df)
    c.execute("CREATE TABLE posts AS SELECT * FROM df")
    return c


def test_removal_rate_wilson(con):
    out = removal_rate(con, "AmItheAsshole")
    assert out["n"] == 25  # 10 survived + 15 removed_mod; author_deleted excluded
    assert out["removed"] == 15
    assert out["rate"] == pytest.approx(0.6)
    assert 0 < out["ci_low"] < out["rate"] < out["ci_high"] < 1
    assert out["ci_low"] == pytest.approx(0.4074, abs=1e-3)
    assert out["ci_high"] == pytest.approx(0.7660, abs=1e-3)


def test_removal_rate_where_and_window(con):
    out = removal_rate(con, "AmItheAsshole", where_sql="length(selftext) > 100")
    assert 0 < out["n"] <= 25
    # window entirely before the fixture data -> empty, None rate
    out = removal_rate(con, "AmItheAsshole", window=("2020-01-01", "2020-01-31"))
    assert out == {"n": 0, "removed": 0, "rate": None, "ci_low": None, "ci_high": None}
    # window spanning the fixture recovers everything (end date inclusive)
    out = removal_rate(con, "AmItheAsshole", window=("2026-08-01", "2026-08-25"))
    assert out["n"] == 25


def test_validator_rejects_semicolon():
    with pytest.raises(StatsQueryError, match="semicolon"):
        validate_where_sql("1=1; DROP TABLE posts")


def test_validator_rejects_subquery():
    with pytest.raises(StatsQueryError, match="not allowed"):
        validate_where_sql("title IN (SELECT title FROM posts)")


def test_validator_rejects_unknown_column():
    with pytest.raises(StatsQueryError, match="score"):
        validate_where_sql("score > 100")


def test_validator_rejects_comments_and_empty():
    with pytest.raises(StatsQueryError):
        validate_where_sql("over_18 -- sneaky")
    with pytest.raises(StatsQueryError):
        validate_where_sql("   ")


def test_validator_accepts_reasonable_fragments():
    for frag in (
        "over_18",
        "length(selftext) > 500 AND NOT is_self",
        "lower(title) LIKE '%aita%'",
        "regexp_matches(title, 'help') OR link_flair_text IS NULL",
        "title ILIKE ?",
    ):
        assert validate_where_sql(frag) == frag


def test_removal_rate_bad_sql_raises(con):
    with pytest.raises(StatsQueryError, match="failed to execute"):
        removal_rate(con, "AmItheAsshole", where_sql="length(title")


def test_compare(con):
    out = compare(con, "AmItheAsshole", "length(selftext) > 1000")
    assert set(out) == {"when_true", "when_false", "lift", "n_true", "n_false"}
    assert out["n_true"] + out["n_false"] == 25
    assert out["n_true"] == out["when_true"]["n"]
    if out["when_true"]["rate"] is not None and out["when_false"]["rate"]:
        assert out["lift"] == pytest.approx(
            out["when_true"]["rate"] / out["when_false"]["rate"], abs=1e-3
        )


@pytest.fixture(scope="module")
def retriever(records) -> tuple[TfidfRetriever, pd.DataFrame]:
    rows = [
        {
            "id": r.id,
            "text": r.title + "\n\n" + r.selftext,
            "created_utc": r.created_utc,
            "label": r.label,
        }
        for r in records
        if r.label in ("survived", "removed_mod") and r.text_available
    ]
    df = pd.DataFrame(rows)
    return TfidfRetriever().fit(df), df


def test_query_self_similarity(retriever):
    r, df = retriever
    row = df.iloc[0]
    results = r.query(row["text"], k=5)
    assert results and results[0]["id"] == row["id"]
    assert results[0]["score"] > 0.9
    assert set(results[0]) == {"id", "score", "label", "created_utc"}
    assert len(results) <= config.RETRIEVAL_TOP_K


def test_temporal_mask(retriever):
    r, df = retriever
    earliest = int(df["created_utc"].min())
    assert r.query(df.iloc[0]["text"], before_utc=earliest) == []
    cutoff = int(df["created_utc"].median())
    for res in r.query(df.iloc[0]["text"], before_utc=cutoff):
        assert res["created_utc"] < cutoff


def test_neighbor_stats(retriever):
    r, df = retriever
    results = r.query(df.iloc[0]["text"], k=10)
    stats = neighbor_stats(results)
    assert stats["k"] == len(results)
    assert 0 <= stats["removed"] <= stats["k"]
    assert 0 <= stats["rate"] <= 1
    assert neighbor_stats([]) == {"k": 0, "removed": 0, "rate": None}


def test_save_load_roundtrip(retriever, tmp_path):
    r, df = retriever
    path = r.save(tmp_path / "index" / "tfidf.joblib")
    r2 = TfidfRetriever.load(path)
    assert r2.query(df.iloc[0]["text"], k=3) == r.query(df.iloc[0]["text"], k=3)
