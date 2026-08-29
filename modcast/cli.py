"""ModCast command line: ingest -> index -> induce -> eval -> predict."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from modcast import config
from modcast.store import Store

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def ingest() -> None:
    """Load every cached raw file into duckdb (idempotent upsert)."""
    store = Store()
    for path in sorted(config.RAW_DIR.glob("*.jsonl.gz")):
        counts = store.ingest_jsonl_gz(path)
        typer.echo(f"{path.name}: {counts}")
    typer.echo(f"totals: {store.counts()}")


@app.command()
def index() -> None:
    """Build one TF-IDF retriever per subreddit over the index window."""
    from modcast.retrieval import TfidfRetriever
    from modcast.stats import _iso_to_epoch  # shared epoch conversion

    store = Store()
    for sub in config.SUBREDDITS:
        w = config.index_window(sub)
        start = _iso_to_epoch(w[0])
        end = _iso_to_epoch(w[1], exclusive_end=True)
        df = store.query(
            """
            SELECT id, title || chr(10) || chr(10) || selftext AS text, created_utc, label
            FROM posts
            WHERE subreddit = ? AND label IN ('survived', 'removed_mod')
              AND text_available AND created_utc >= ? AND created_utc < ?
            ORDER BY id
            """,
            [sub, start, end],
        ).df()
        if df.empty:
            typer.echo(f"{sub}: no data in index window — skipped")
            continue
        path = TfidfRetriever().fit(df).save(config.DATA_DIR / "index" / f"{sub}.joblib")
        typer.echo(f"{sub}: indexed {len(df)} posts -> {path}")


@app.command()
def induce(
    sub: list[str] = typer.Option(None, help="Subreddits (default: all configured)"),
    model: str = typer.Option(None),
    effort: str = typer.Option("high"),
) -> None:
    """Induce a verified rulebook per subreddit (writes rulebooks/{sub}.md)."""
    from modcast.induce import induce as run_induce
    from modcast.llm import new_run_id

    store = Store()
    run_id = new_run_id("induce")
    for s in sub or config.SUBREDDITS:
        res = run_induce(store.con, s, run_id=run_id, model=model, effort=effort)
        typer.echo(
            f"{s}: kept {len(res.kept)}, dropped {len(res.dropped)} -> {res.path} "
            f"({res.input_tokens} in / {res.output_tokens} out tokens)"
        )
        for d in res.dropped:
            typer.echo(f"  dropped: {d['rule'][:70]} — {d['why']}")


@app.command()
def eval(
    with_llm: bool = typer.Option(False, help="Also evaluate LLM one-shot + agent on the seeded subsample"),
    llm_posts: int = typer.Option(config.LLM_EVAL_POSTS_PER_SUB, help="LLM subsample size per subreddit"),
    model: str = typer.Option(None),
    effort: str = typer.Option("high"),
    no_rulebook: bool = typer.Option(False, help="Ablation: run the agent without rulebooks"),
) -> None:
    """Run the harness. Cheap predictors on the full test set; LLM predictors
    (and the cheap ones again, for same-case comparison) on the subsample."""
    from modcast import evaluate as E
    from modcast.baselines import BaseRatePredictor, LogisticPredictor

    store = Store()
    train, test = E.build_split(store.con)
    typer.echo(f"train={len(train)} test={len(test)}")

    cheap = [BaseRatePredictor(), LogisticPredictor()]
    report = E.run_eval(cheap, train, test)
    typer.echo(json.dumps({k: v["overall"] for k, v in report["predictors"].items()}, indent=1))

    if with_llm:
        from modcast.llm import new_run_id
        from modcast.llm_predictors import AgentPredictor, OneShotPredictor

        rng = np.random.default_rng(config.RANDOM_SEED + 1)
        subsample = []
        for s in sorted({r.subreddit for r in test}):
            pool = [r for r in test if r.subreddit == s]
            idx = rng.permutation(len(pool))[:llm_posts]
            subsample.extend(pool[i] for i in sorted(idx))
        run_id = new_run_id("eval")
        typer.echo(f"LLM eval on {len(subsample)} posts, run_id={run_id}")
        agent = AgentPredictor(store.con, run_id=run_id, model=model, effort=effort,
                               use_rulebook=not no_rulebook)
        preds = cheap + [OneShotPredictor(run_id=run_id, model=model), agent]
        out_dir = config.RESULTS_DIR / ("llm_norulebook" if no_rulebook else "llm")
        report = E.run_eval(preds, train, subsample, out_dir=out_dir)
        typer.echo(json.dumps({k: v["overall"] for k, v in report["predictors"].items()}, indent=1))
        spend = {"input_tokens": sum(f.input_tokens for f in agent.forecasts.values()),
                 "output_tokens": sum(f.output_tokens for f in agent.forecasts.values()),
                 "mean_turns": float(np.mean([f.turns for f in agent.forecasts.values()])) if agent.forecasts else None}
        (out_dir / "agent_spend.json").write_text(json.dumps(spend, indent=1))
        typer.echo(f"agent spend: {spend}")


@app.command()
def predict(
    sub: str = typer.Option(...),
    title: str = typer.Option(None),
    body: str = typer.Option("", help="Post body (or use --body-file)"),
    body_file: Path = typer.Option(None),
    post_id: str = typer.Option(None, help="Forecast a real archived post instead"),
    model: str = typer.Option(None),
    effort: str = typer.Option("high"),
) -> None:
    """Preflight one post: probability + verified risk factors + fixes."""
    from modcast import agent as A
    from modcast import stats as S
    from modcast.llm import new_run_id
    from modcast.report import render
    from modcast.retrieval import TfidfRetriever
    from modcast.schema import PostRecord, normalize
    from modcast.subrules import rules_digest

    store = Store()
    if post_id:
        raw_row = store.query("SELECT * FROM posts WHERE id = ?", [post_id]).df()
        if raw_row.empty:
            raise typer.BadParameter(f"post {post_id} not in db")
        d = raw_row.iloc[0].to_dict()
        record = PostRecord(**{k: d[k] for k in PostRecord.__dataclass_fields__})
    else:
        if title is None:
            raise typer.BadParameter("--title required (or --post-id)")
        if body_file:
            body = body_file.read_text()
        record = normalize({
            "id": f"draft-{int(time.time())}", "subreddit": sub,
            "created_utc": int(time.time()), "title": title, "selftext": body,
            "is_self": True,
        })
    rb_path = config.RULEBOOK_DIR / f"{sub}.md"
    ctx = A.AgentContext(
        con=store.con,
        retriever=TfidfRetriever.load(config.DATA_DIR / "index" / f"{sub}.joblib"),
        subreddit=sub,
        window=config.index_window(sub),
    )
    run_id = new_run_id("predict")
    fc = A.forecast(
        ctx, record, run_id=run_id,
        rulebook=rb_path.read_text() if rb_path.exists() else "",
        published_rules=rules_digest(sub), model=model, effort=effort,
    )
    base = S.removal_rate(store.con, sub, window=ctx.window)["rate"]
    text = render(record, fc, base_rate=base)
    out = config.RESULTS_DIR / "reports" / f"{record.id}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    typer.echo(text)
    typer.echo(f"[saved {out} | trajectory results/trajectories/{run_id}/]")


if __name__ == "__main__":
    app()
