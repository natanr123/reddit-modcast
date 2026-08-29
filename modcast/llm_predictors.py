"""LLM predictors implementing the same Predictor protocol as the baselines.

OneShotPredictor is the REQUIRED fair baseline: the same model, one direct
prompt, no tools, no corpus. AgentPredictor is the full ModCast pipeline.
Both are evaluated on identical cases by the same harness.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import duckdb
import numpy as np

from modcast import agent as A
from modcast.config import RULEBOOK_DIR, DATA_DIR, index_window
from modcast.llm import LLMSession, text_of
from modcast.retrieval import TfidfRetriever
from modcast.schema import PostRecord
from modcast.subrules import rules_digest

CONCURRENCY = int(os.environ.get("MODCAST_CONCURRENCY", "4"))

ONESHOT_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "p_removed": {"type": "number", "minimum": 0, "maximum": 1},
            "reasoning": {"type": "string"},
        },
        "required": ["p_removed", "reasoning"],
        "additionalProperties": False,
    },
}


class OneShotPredictor:
    """Baseline: one direct prompt to the same model — what people do today."""

    name = "llm_oneshot"

    def __init__(self, run_id: str, model: str | None = None):
        self.run_id = run_id
        self.model = model

    def _one(self, r: PostRecord) -> float:
        session = LLMSession(
            run_id=self.run_id, name=f"oneshot-{r.id}", effort="high",
            **({"model": self.model} if self.model else {}),
        )
        prompt = (
            f"Estimate the probability (0..1) that this Reddit post will be removed "
            f"by moderation (mods, automod, or admins — not author deletion) within "
            f"36 hours of being posted in r/{r.subreddit}.\n\n"
            f"TITLE: {r.title}\nBODY:\n{r.selftext[:4000]}\n\n"
            f"Respond with JSON: p_removed and one-sentence reasoning."
        )
        try:
            resp = session.step(prompt, max_tokens=2000, output_format=ONESHOT_SCHEMA)
            data = json.loads(text_of(resp))
            return float(min(1.0, max(0.0, data["p_removed"])))
        except Exception:
            return 0.5  # a baseline that errors answers with maximum uncertainty

    def predict_proba(self, records: list[PostRecord]) -> np.ndarray:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            return np.array(list(ex.map(self._one, records)))


class AgentPredictor:
    """The full ModCast agent: dossier -> tool investigation -> verified forecast."""

    name = "modcast_agent"

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        run_id: str,
        model: str | None = None,
        effort: str = "high",
        use_rulebook: bool = True,   # ablation switch for the changelog
    ):
        self.con = con
        self.run_id = run_id
        self.model = model
        self.effort = effort
        self.use_rulebook = use_rulebook
        self._sub_cache: dict[str, tuple[TfidfRetriever, str, str]] = {}
        self.forecasts: dict[str, A.Forecast] = {}

    def _sub_assets(self, subreddit: str) -> tuple[TfidfRetriever, str, str]:
        if subreddit not in self._sub_cache:
            retriever = TfidfRetriever.load(DATA_DIR / "index" / f"{subreddit}.joblib")
            rb_path = RULEBOOK_DIR / f"{subreddit}.md"
            rulebook = rb_path.read_text() if (self.use_rulebook and rb_path.exists()) else ""
            self._sub_cache[subreddit] = (retriever, rulebook, rules_digest(subreddit))
        return self._sub_cache[subreddit]

    def _one(self, r: PostRecord) -> float:
        retriever, rulebook, rules = self._sub_assets(r.subreddit)
        ctx = A.AgentContext(
            con=self.con.cursor(),  # duckdb: one cursor per thread
            retriever=retriever,
            subreddit=r.subreddit,
            window=index_window(r.subreddit),
        )
        try:
            fc = A.forecast(
                ctx, r, run_id=self.run_id, rulebook=rulebook,
                published_rules=rules, model=self.model, effort=self.effort,
            )
            self.forecasts[r.id] = fc
            return fc.p_removed
        except Exception as e:
            print(f"[agent] {r.id} failed: {type(e).__name__}: {e}")
            return 0.5

    def predict_proba(self, records: list[PostRecord]) -> np.ndarray:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            return np.array(list(ex.map(self._one, records)))
