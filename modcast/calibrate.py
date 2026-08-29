"""Isotonic calibration wrapper for any Predictor."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression

from modcast.config import RANDOM_SEED
from modcast.baselines import Predictor
from modcast.schema import PostRecord


class CalibratedPredictor:
    """Wraps a Predictor; isotonic regression fit on a held-out slice of train.

    The split is deterministic (RANDOM_SEED). The base predictor, if it has
    a `fit`, is trained only on the non-validation slice.
    """

    def __init__(self, base: Predictor, val_fraction: float = 0.2, seed: int = RANDOM_SEED) -> None:
        self.base = base
        self.val_fraction = val_fraction
        self.seed = seed
        self.name = f"{base.name}+isotonic"
        self.iso: IsotonicRegression | None = None

    def fit(self, records: list[PostRecord], labels: Sequence[int]) -> "CalibratedPredictor":
        y = np.asarray(labels, dtype=int)
        idx = np.random.default_rng(self.seed).permutation(len(records))
        n_val = max(1, int(len(records) * self.val_fraction))
        val_idx, fit_idx = idx[:n_val], idx[n_val:]
        if hasattr(self.base, "fit"):
            self.base.fit([records[i] for i in fit_idx], y[fit_idx])
        p_val = np.asarray(self.base.predict_proba([records[i] for i in val_idx]), dtype=float)
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_val, y[val_idx])
        return self

    def predict_proba(self, records: list[PostRecord]) -> np.ndarray:
        assert self.iso is not None, "fit() first"
        return self.iso.predict(np.asarray(self.base.predict_proba(records), dtype=float))
