"""Run-to-run stability: K independent forecasts of the same fixed posts.

The prediction cache is deliberately bypassed (agent.forecast is called
directly), so every repetition is a fresh investigation. Output:
results/stability.json + a printed summary of per-post spread.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from modcast import agent as A
from modcast import config
from modcast import evaluate as E
from modcast.retrieval import load_retriever
from modcast.store import Store
from modcast.subrules import rules_digest

K = 5
PER_SUB = 5

store = Store(read_only=True)
_, test = E.build_split(store.con)
rng = np.random.default_rng(config.RANDOM_SEED + 2)
posts = []
for s in sorted({r.subreddit for r in test}):
    pool = [r for r in test if r.subreddit == s]
    idx = rng.permutation(len(pool))[:PER_SUB]
    posts.extend(pool[i] for i in sorted(idx))
print(f"stability: {len(posts)} posts x {K} reps")

retrievers = {s: load_retriever(s) for s in config.SUBREDDITS}
rules = {s: rules_digest(s) for s in config.SUBREDDITS}


def one(task):
    rep, r = task
    ctx = A.AgentContext(
        con=store.con.cursor(), retriever=retrievers[r.subreddit],
        subreddit=r.subreddit, window=config.index_window(r.subreddit),
    )
    try:
        fc = A.forecast(ctx, r, run_id=f"stability-rep{rep}", rulebook="",
                        published_rules=rules[r.subreddit])
        return rep, r.id, fc.p_removed
    except Exception as e:
        print(f"fail rep{rep} {r.id}: {type(e).__name__}: {e}")
        return rep, r.id, None


tasks = [(rep, r) for rep in range(K) for r in posts]
with ThreadPoolExecutor(max_workers=8) as ex:
    results = list(ex.map(one, tasks))

mat: dict[str, list] = {}
for rep, pid, p in results:
    if p is not None:
        mat.setdefault(pid, []).append(p)

rows = []
for pid, ps in mat.items():
    if len(ps) < 2:
        continue
    arr = np.array(ps)
    pw = [abs(a - b) for i, a in enumerate(arr) for b in arr[i + 1:]]
    rows.append({"id": pid, "k": len(ps), "mean_p": round(float(arr.mean()), 4),
                 "std": round(float(arr.std()), 4),
                 "spread": round(float(arr.max() - arr.min()), 4),
                 "mean_pairwise_diff": round(float(np.mean(pw)), 4)})

summary = {
    "posts": len(rows), "reps": K,
    "median_std": round(float(np.median([r["std"] for r in rows])), 4),
    "median_spread": round(float(np.median([r["spread"] for r in rows])), 4),
    "mean_pairwise_diff": round(float(np.mean([r["mean_pairwise_diff"] for r in rows])), 4),
    "max_spread": round(float(max(r["spread"] for r in rows)), 4),
}
out = config.RESULTS_DIR / "stability.json"
out.write_text(json.dumps({"summary": summary, "per_post": rows}, indent=1))
print("STABILITY SUMMARY:", json.dumps(summary))
print(f"written {out}")
