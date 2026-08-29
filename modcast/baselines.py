"""Baseline predictors implementing the shared Predictor protocol.

Every predictor (baselines here, LLM predictors elsewhere) exposes
`name` and `predict_proba(records) -> P(removed_mod)`; the eval harness
in modcast.evaluate treats them interchangeably.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from modcast.schema import PostRecord


@runtime_checkable
class Predictor(Protocol):
    name: str

    def predict_proba(self, records: list[PostRecord]) -> np.ndarray:
        """Return P(removed_mod) per record, shape (n,)."""
        ...


def _featurize(records: list[PostRecord]) -> np.ndarray:
    from modcast.features import featurize_records  # lazy: module built separately

    X = featurize_records(records)
    if isinstance(X, tuple):  # tolerate (X, feature_names) shape
        X = X[0]
    return np.asarray(X, dtype=float)


class BaseRatePredictor:
    """Predicts each subreddit's train-window removal rate (global fallback)."""

    name = "base_rate"

    def __init__(self) -> None:
        self.rates: dict[str, float] = {}
        self.global_rate: float = 0.5

    def fit(self, records: list[PostRecord], labels: Sequence[int]) -> "BaseRatePredictor":
        y = np.asarray(labels, dtype=float)
        if len(y):
            self.global_rate = float(y.mean())
        subs = np.array([r.subreddit for r in records])
        self.rates = {s: float(y[subs == s].mean()) for s in np.unique(subs)}
        return self

    def predict_proba(self, records: list[PostRecord]) -> np.ndarray:
        return np.array([self.rates.get(r.subreddit, self.global_rate) for r in records])


class LogisticPredictor:
    """Scaled logistic regression over modcast.features at-post-time features."""

    name = "logistic"

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(class_weight="balanced", max_iter=2000)

    def fit(self, records: list[PostRecord], labels: Sequence[int]) -> "LogisticPredictor":
        X = self.scaler.fit_transform(_featurize(records))
        self.clf.fit(X, np.asarray(labels, dtype=int))
        return self

    def predict_proba(self, records: list[PostRecord]) -> np.ndarray:
        X = self.scaler.transform(_featurize(records))
        return self.clf.predict_proba(X)[:, 1]
