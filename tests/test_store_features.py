"""Store + features against the 30-post AITA fixture."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from modcast.features import FEATURE_NAMES, extract, featurize_records
from modcast.schema import normalize
from modcast.store import Store

FIXTURE = Path(__file__).parent / "fixtures" / "sample_aita.json"


@pytest.fixture()
def raws() -> list[dict]:
    return json.loads(FIXTURE.read_text())


@pytest.fixture()
def store(tmp_path: Path, raws: list[dict]) -> Store:
    s = Store(tmp_path / "test.duckdb")
    s.ingest_raw(raws)
    yield s
    s.close()


def test_ingest_counts(store: Store) -> None:
    c = store.counts()
    assert c[("survived", True)] == 10
    assert c[("removed_mod", True)] == 10
    assert c[("removed_mod", False)] == 5
    assert c[("author_deleted", True)] == 5
    assert sum(c.values()) == 30
    assert store.counts("AmItheAsshole") == c
    assert store.counts("nosuchsub") == {}


def test_ingest_is_upsert(store: Store, raws: list[dict]) -> None:
    result = store.ingest_raw(raws)  # second pass: same ids, no duplicates
    assert result["total"] == 30
    assert store.query("SELECT count(*) FROM posts").fetchone()[0] == 30


def test_ingest_jsonl_gz(tmp_path: Path, raws: list[dict]) -> None:
    gz = tmp_path / "posts.jsonl.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as f:
        for raw in raws:
            f.write(json.dumps(raw) + "\n")
    s = Store(tmp_path / "gz.duckdb")
    result = s.ingest_jsonl_gz(gz)
    assert result["total"] == 30
    assert result["removed_mod"] == 15
    assert sum(s.counts().values()) == 30
    s.close()


def test_query_passthrough(store: Store) -> None:
    rows = store.query(
        "SELECT id FROM posts WHERE label = ? ORDER BY created_utc", ["survived"]
    ).fetchall()
    assert len(rows) == 10


def test_features_shape_and_no_nans(raws: list[dict]) -> None:
    records = [normalize(r) for r in raws]
    df = featurize_records(records)
    assert list(df.columns) == FEATURE_NAMES
    assert len(df) == 30
    assert not df.isna().any().any()
    assert df.map(lambda v: isinstance(v, float)).all().all()


def test_features_deterministic(raws: list[dict]) -> None:
    records = [normalize(r) for r in raws]
    assert featurize_records(records).equals(featurize_records(records))
    one = records[0]
    assert list(extract(one)) == FEATURE_NAMES
    assert extract(one) == extract(one)


def test_feature_values_spot_check(raws: list[dict]) -> None:
    rec = normalize(raws[0])  # AITA post with full text
    f = extract(rec)
    assert f["title_len"] == float(len(rec.title))
    assert f["title_bracket_tag"] == 1.0  # title starts with AITA(H)
    assert f["body_len"] > 0
    assert 0.0 < f["body_first_person_ratio"] < 1.0
    assert f["is_self"] == 1.0
    assert 0 <= f["created_hour_utc"] <= 23
    assert 0 <= f["created_weekday"] <= 6
    # textless removed posts still featurize cleanly
    empty = next(r for r in map(normalize, raws) if not r.text_available)
    fe = extract(empty)
    assert fe["body_len"] == 0.0
    assert fe["title_body_ratio"] == 0.0
    assert all(v == v for v in fe.values())  # no NaN
