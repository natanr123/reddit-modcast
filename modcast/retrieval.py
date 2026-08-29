"""TF-IDF nearest-neighbor retrieval over indexed posts.

Agent-facing: given a draft post's text, return the most similar historical
posts (with their moderation outcomes) so the LLM can reason from precedent.
The caller builds the fit dataframe; text is `title + "\n\n" + selftext`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from modcast import config

DEFAULT_INDEX_PATH = config.DATA_DIR / "index" / "tfidf.joblib"


class TfidfRetriever:
    """TF-IDF index with a temporal mask (never surfaces future neighbors)."""

    def __init__(self) -> None:
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.ids: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.created: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "TfidfRetriever":
        """Fit on df with columns id, text, created_utc, label."""
        self.vectorizer = TfidfVectorizer(
            max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform(df["text"].astype(str).tolist())
        self.ids = df["id"].to_numpy()
        self.labels = df["label"].to_numpy()
        self.created = df["created_utc"].to_numpy(dtype=np.int64)
        return self

    def query(
        self,
        text: str,
        k: int = config.RETRIEVAL_TOP_K,
        before_utc: int | None = None,
    ) -> list[dict[str, Any]]:
        """Top-k cosine neighbors as {id, score, label, created_utc}.

        With before_utc set, neighbors created at or after it are excluded.
        Zero-similarity posts are never returned.
        """
        if self.vectorizer is None:
            raise RuntimeError("retriever is not fitted; call fit() or load() first.")
        scores = (self.matrix @ self.vectorizer.transform([text]).T).toarray().ravel()
        mask = scores > 0
        if before_utc is not None:
            mask &= self.created < before_utc
        idx = np.flatnonzero(mask)
        idx = idx[np.argsort(scores[idx])[::-1][:k]]
        return [
            {
                "id": str(self.ids[i]),
                "score": round(float(scores[i]), 4),
                "label": str(self.labels[i]),
                "created_utc": int(self.created[i]),
            }
            for i in idx
        ]

    def save(self, path: str | Path = DEFAULT_INDEX_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_INDEX_PATH) -> "TfidfRetriever":
        return joblib.load(Path(path))


def neighbor_stats(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Removal rate among retrieved neighbors: {k, removed, rate}."""
    k = len(results)
    removed = sum(1 for r in results if r["label"] == "removed_mod")
    return {"k": k, "removed": removed, "rate": round(removed / k, 4) if k else None}


class DenseRetriever:
    """Embedding-based sibling of TfidfRetriever (same interface).

    Matches on meaning rather than shared words; the ablation axis is
    config.RETRIEVER_KIND. Requires the [embed] extra (sentence-transformers);
    the encoder loads lazily so importing this module stays cheap.
    """

    def __init__(self, model_name: str | None = None):
        from modcast.config import DENSE_MODEL

        self.model_name = model_name or DENSE_MODEL
        self._encoder = None
        self.matrix: np.ndarray | None = None
        self.ids: list[str] = []
        self.labels: list[str] = []
        self.created: np.ndarray | None = None

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name)
        return np.asarray(
            self._encoder.encode(texts, batch_size=256, normalize_embeddings=True,
                                 show_progress_bar=False),
            dtype=np.float32,
        )

    def fit(self, df: pd.DataFrame) -> "DenseRetriever":
        self.ids = df["id"].tolist()
        self.labels = df["label"].tolist()
        self.created = df["created_utc"].to_numpy()
        self.matrix = self._encode(df["text"].tolist())
        return self

    def query(self, text: str, k: int = config.RETRIEVAL_TOP_K,
              before_utc: int | None = None) -> list[dict[str, Any]]:
        if self.matrix is None:
            raise RuntimeError("retriever is not fitted; call fit() or load() first.")
        scores = self.matrix @ self._encode([text])[0]
        mask = np.ones(len(scores), dtype=bool)
        if before_utc is not None:
            mask &= self.created < before_utc
        idx = np.flatnonzero(mask)
        top = idx[np.argsort(-scores[idx])][:k]
        return [{"id": self.ids[i], "score": round(float(scores[i]), 4),
                 "label": self.labels[i], "created_utc": int(self.created[i])} for i in top]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, matrix=self.matrix, ids=np.array(self.ids),
                            labels=np.array(self.labels), created=self.created,
                            model=np.array([self.model_name]))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "DenseRetriever":
        data = np.load(str(path), allow_pickle=False)
        r = cls(model_name=str(data["model"][0]))
        r.matrix = data["matrix"]
        r.ids = data["ids"].tolist()
        r.labels = data["labels"].tolist()
        r.created = data["created"]
        return r


def load_retriever(subreddit: str):
    """The configured retriever for a subreddit (config.RETRIEVER_KIND)."""
    from modcast.config import DATA_DIR, RETRIEVER_KIND

    if RETRIEVER_KIND == "dense":
        return DenseRetriever.load(DATA_DIR / "index" / f"{subreddit}.dense.npz")
    return TfidfRetriever.load(DATA_DIR / "index" / f"{subreddit}.joblib")
