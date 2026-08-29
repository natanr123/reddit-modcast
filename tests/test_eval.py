"""Referee sanity checks: metrics, split determinism, calibration."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from modcast.baselines import BaseRatePredictor
from modcast.calibrate import CalibratedPredictor
from modcast.evaluate import build_split, labels_of, metrics, run_eval
from modcast.schema import Label, PostRecord, normalize

FIXTURE = Path(__file__).parent / "fixtures" / "sample_aita.json"
FIXTURE_WINDOW = ("2026-08-20", "2026-08-20")  # all 30 fixture posts land here


def fixture_records() -> list[PostRecord]:
    return [normalize(raw) for raw in json.loads(FIXTURE.read_text())]


def fixture_con() -> duckdb.DuckDBPyConnection:
    df = pd.DataFrame([r.to_dict() for r in fixture_records()])
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE posts AS SELECT * FROM df")
    return con


def test_perfect_predictor_metrics() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    m = metrics(y, y.astype(float))
    assert m["brier"] < 0.01
    assert m["auc"] == 1.0
    assert m["n"] == 500


def test_constant_predictor_metrics() -> None:
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.3).astype(int)
    br = y.mean()
    m = metrics(y, np.full(len(y), br))
    assert m["auc"] == 0.5
    assert m["brier"] == pytest.approx(br * (1 - br), abs=1e-9)
    assert m["base_rate"] == pytest.approx(br)


def test_ece_perfectly_calibrated() -> None:
    centers = np.arange(0.05, 1.0, 0.10)
    p = np.repeat(centers, 200)
    y = np.concatenate([
        np.r_[np.ones(round(200 * c)), np.zeros(200 - round(200 * c))] for c in centers
    ])
    assert metrics(y, p)["ece"] < 1e-9


def test_build_split_deterministic_and_filtered() -> None:
    con = fixture_con()
    kw = dict(train_window=FIXTURE_WINDOW, test_window=FIXTURE_WINDOW, posts_per_sub=5)
    train1, test1 = build_split(con, **kw)
    train2, test2 = build_split(con, **kw)
    assert [r.id for r in train1] == [r.id for r in train2]
    assert [r.id for r in test1] == [r.id for r in test2]
    # eligibility: only survived/removed_mod with text (fixture: 10 + 10)
    assert len(train1) == 20
    assert all(r.text_available for r in train1)
    assert {r.label for r in train1} <= {Label.SURVIVED.value, Label.REMOVED_MOD.value}
    assert len(test1) == 5  # per-sub cap applied
    # a different seed still samples only eligible ids
    _, test3 = build_split(con, seed=1, **kw)
    assert {r.id for r in test3} <= {r.id for r in train1}


def test_run_eval_writes_reports(tmp_path: Path) -> None:
    _, records = build_split(
        fixture_con(), train_window=FIXTURE_WINDOW, test_window=FIXTURE_WINDOW, posts_per_sub=20
    )
    report = run_eval([BaseRatePredictor()], records, records, out_dir=tmp_path)
    overall = report["predictors"]["base_rate"]["overall"]
    assert overall["n"] == 20
    assert overall["base_rate"] == pytest.approx(0.5)
    assert (tmp_path / "eval_latest.json").exists()
    assert (tmp_path / "eval_latest.md").exists()
    assert (tmp_path / "figures" / "reliability_base_rate.png").exists()
    assert json.loads((tmp_path / "eval_latest.json").read_text()) == report


class _MiscalibratedPredictor:
    """Protocol-shaped stub: squashes true probs (its 'records') toward 0.5."""

    name = "stub"

    def predict_proba(self, records: list) -> np.ndarray:
        return 0.5 + 0.2 * (np.asarray(records, dtype=float) - 0.5)


def test_calibrated_predictor_improves_ece() -> None:
    rng = np.random.default_rng(2)
    true_p = rng.random(4000)
    y = (rng.random(4000) < true_p).astype(int)
    records = list(true_p)  # stub predictor reads records as its raw scores
    base = _MiscalibratedPredictor()
    cal = CalibratedPredictor(base, val_fraction=0.5)
    cal.fit(records, y)
    assert cal.name == "stub+isotonic"
    p_cal = cal.predict_proba(records)
    assert metrics(y, p_cal)["ece"] < metrics(y, base.predict_proba(records))["ece"]
    # deterministic refit
    cal2 = CalibratedPredictor(_MiscalibratedPredictor(), val_fraction=0.5)
    cal2.fit(records, y)
    assert np.array_equal(p_cal, cal2.predict_proba(records))


def test_labels_of() -> None:
    recs = [r for r in fixture_records() if r.label != Label.AUTHOR_DELETED.value]
    y = labels_of(recs)
    assert set(y) == {0, 1}
    assert y.sum() == 15  # 10 with text + 5 textless removed_mod
