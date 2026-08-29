"""Evaluation harness: the referee. Deterministic splits, metrics, reports.

`build_split` pulls train/test records from the duckdb `posts` table using
the config windows; `run_eval` scores any list of Predictors on the same
frozen test set and writes results/eval_latest.{json,md} plus reliability
plots. Only label IN (survived, removed_mod) AND text_available posts count.
"""
from __future__ import annotations

import dataclasses
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import duckdb
import numpy as np

from modcast import config
from modcast.baselines import Predictor
from modcast.schema import Label, PostRecord

FIELDS = [f.name for f in dataclasses.fields(PostRecord)]
ELIGIBLE_LABELS = (Label.SURVIVED.value, Label.REMOVED_MOD.value)


def _epoch(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def labels_of(records: Sequence[PostRecord]) -> np.ndarray:
    return np.array([int(r.label == Label.REMOVED_MOD.value) for r in records])


def _fetch(con: duckdb.DuckDBPyConnection, start: str, end: str) -> list[PostRecord]:
    """Eligible posts with created_utc in [start, end] (end inclusive), ORDER BY id."""
    rows = con.execute(
        f"SELECT {', '.join(FIELDS)} FROM posts"
        " WHERE created_utc >= ? AND created_utc < ?"
        " AND label IN (?, ?) AND text_available ORDER BY id",
        [_epoch(start), _epoch(end) + 86400, *ELIGIBLE_LABELS],
    ).fetchall()
    return [PostRecord(**dict(zip(FIELDS, row))) for row in rows]


def build_split(
    con: duckdb.DuckDBPyConnection,
    train_window: tuple[str, str] = (config.INDEX_START, config.INDEX_END),
    test_window: tuple[str, str] = (config.TEST_START, config.TEST_END),
    posts_per_sub: int = config.EVAL_POSTS_PER_SUB,
    seed: int = config.RANDOM_SEED,
) -> tuple[list[PostRecord], list[PostRecord]]:
    """Train = all eligible index-window posts; test = seeded per-sub sample."""
    train = _fetch(con, *train_window)
    pool = _fetch(con, *test_window)
    rng = np.random.default_rng(seed)
    test: list[PostRecord] = []
    for sub in sorted({r.subreddit for r in pool}):
        sub_pool = [r for r in pool if r.subreddit == sub]  # already id-sorted
        idx = rng.permutation(len(sub_pool))[:posts_per_sub]
        test.extend(sub_pool[i] for i in sorted(idx))
    return train, test


def metrics(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    pc = np.clip(p, 1e-12, 1 - 1e-12)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    return {
        "brier": float(brier_score_loss(y, p)),
        "auc": auc,
        "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
        "ece": _ece(y, p),
        "n": int(len(y)),
        "base_rate": float(y.mean()) if len(y) else float("nan"),
    }


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def reliability_plot(name: str, y: np.ndarray, p: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = np.minimum((np.asarray(p) * 10).astype(int), 9)
    xs, ys, ns = [], [], []
    for b in range(10):
        mask = idx == b
        if mask.any():
            xs.append(p[mask].mean())
            ys.append(y[mask].mean())
            ns.append(int(mask.sum()))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.plot(xs, ys, "o-", color="#d33")
    for x_, y_, n_ in zip(xs, ys, ns):
        ax.annotate(str(n_), (x_, y_), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="mean predicted P(removed)",
           ylabel="observed removal rate", title=f"Reliability: {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name)


def run_eval(
    predictors: Sequence[Predictor],
    train_records: list[PostRecord],
    test_records: list[PostRecord],
    out_dir: Path = config.RESULTS_DIR,
) -> dict:
    """Fit (where supported), score, plot, and report every predictor."""
    y_train = labels_of(train_records)
    y_test = labels_of(test_records)
    report: dict = {
        "seed": config.RANDOM_SEED,
        "train": {"n": int(len(y_train)), "base_rate": float(y_train.mean()) if len(y_train) else None},
        "test": {"n": int(len(y_test)), "base_rate": float(y_test.mean())},
        "predictors": {},
    }
    subs = sorted({r.subreddit for r in test_records})
    for pred in predictors:
        if hasattr(pred, "fit"):
            pred.fit(train_records, y_train)
        p = np.asarray(pred.predict_proba(test_records), dtype=float)
        fig_path = out_dir / "figures" / f"reliability_{_safe(pred.name)}.png"
        reliability_plot(pred.name, y_test, p, fig_path)
        per_sub = {}
        for sub in subs:
            mask = np.array([r.subreddit == sub for r in test_records])
            per_sub[sub] = metrics(y_test[mask], p[mask])
        report["predictors"][pred.name] = {
            "overall": metrics(y_test, p),
            "per_subreddit": per_sub,
            "figure": str(fig_path.relative_to(out_dir)),
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_latest.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "eval_latest.md").write_text(_markdown(report))
    return report


def _markdown(report: dict) -> str:
    cols = ["brier", "auc", "log_loss", "ece", "n", "base_rate"]

    def row(name: str, m: dict) -> str:
        vals = [f"{m[c]:.4f}" if c != "n" else str(m[c]) for c in cols]
        return f"| {name} | " + " | ".join(vals) + " |"

    lines = [
        "# Eval report (latest)",
        "",
        f"Train: n={report['train']['n']}, base_rate={report['train']['base_rate']}; "
        f"Test: n={report['test']['n']}, base_rate={report['test']['base_rate']:.4f}; "
        f"seed={report['seed']}",
        "",
        "| predictor | " + " | ".join(cols) + " |",
        "|" + "---|" * (len(cols) + 1),
    ]
    for name, res in report["predictors"].items():
        lines.append(row(f"**{name}**", res["overall"]))
        for sub, m in res["per_subreddit"].items():
            lines.append(row(f"{name} / r/{sub}", m))
    return "\n".join(lines) + "\n"


def main() -> None:
    if not config.DB_PATH.exists():
        print(f"skip: no database at {config.DB_PATH} (run the fetch/store pipeline first)")
        return
    from modcast.baselines import BaseRatePredictor, LogisticPredictor

    con = duckdb.connect(str(config.DB_PATH), read_only=True)
    train, test = build_split(con)
    print(f"split: {len(train)} train / {len(test)} test")
    predictors: list[Predictor] = [BaseRatePredictor()]
    try:
        import modcast.features  # noqa: F401

        predictors.append(LogisticPredictor())
    except ImportError:
        print("skip logistic: modcast.features not available yet")
    report = run_eval(predictors, train, test)
    for name, res in report["predictors"].items():
        o = res["overall"]
        print(f"{name}: brier={o['brier']:.4f} auc={o['auc']:.4f} ece={o['ece']:.4f}")
    print(f"wrote {config.RESULTS_DIR / 'eval_latest.json'} and .md")


if __name__ == "__main__":
    main()
