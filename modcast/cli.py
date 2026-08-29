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

app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def _main(ctx: typer.Context) -> None:
    """ModCast — a weather forecast for your Reddit post."""
    if ctx.invoked_subcommand is None:
        interactive()


def _available_subs() -> list[str]:
    return sorted(p.stem for p in (config.DATA_DIR / "index").glob("*.joblib"))


def interactive() -> None:
    """No-command mode: ask what to do, collect inputs, run, repeat."""
    typer.secho("\n⛅ ModCast — a weather forecast for your Reddit post", bold=True)
    while True:
        typer.echo(
            "\nWhat do you want to do?\n"
            "  1. Forecast a draft post (before you post it)\n"
            "  2. Forecast a Reddit post by URL\n"
            "  3. Forecast an archived post by id\n"
            "  4. Quit"
        )
        choice = typer.prompt("Choice", default="1").strip()
        try:
            if choice == "1":
                subs = _available_subs()
                typer.echo(f"Available subreddits: {', '.join(subs)}")
                sub = typer.prompt("Subreddit").strip().removeprefix("r/")
                title = typer.prompt("Title")
                typer.echo("Body (finish with an empty line):")
                body_lines: list[str] = []
                while (line := input()) != "":
                    body_lines.append(line)
                flair = typer.prompt("Flair you will select (Enter for none)", default="").strip() or None
                _predict_or_offer_onboard(sub=sub, title=title, body="\n".join(body_lines),
                        body_file=None, flair=flair, post_id=None, url=None, model=None, effort=None)
            elif choice == "2":
                _predict_or_offer_onboard(sub=None, title=None, body="", body_file=None, flair=None,
                        post_id=None, url=typer.prompt("Reddit post URL").strip(), model=None, effort=None)
            elif choice == "3":
                _predict_or_offer_onboard(sub=None, title=None, body="", body_file=None, flair=None,
                        post_id=typer.prompt("Post id (base36)").strip(), url=None,
                        model=None, effort=None)
            elif choice in ("4", "q", "quit", "exit"):
                raise typer.Exit()
            else:
                typer.secho(f"Unknown choice: {choice}", fg="yellow")
        except typer.Exit:
            raise
        except (typer.BadParameter, Exception) as e:  # keep the loop alive on errors
            typer.secho(f"error: {e}", fg="red")


@app.command()
def onboard(
    sub: str = typer.Argument(..., help="Subreddit to onboard (without r/)"),
    days: int = typer.Option(90, help="How much history to fetch"),
) -> None:
    """Onboard a new subreddit: fetch labeled history, ingest, build its index.

    Posts newer than 48h are skipped (their second-pass removal labels are not
    mature yet). Accuracy on onboarded subs is NOT covered by the measured
    eval — that covers the four curated subreddits.
    """
    import time as _t

    from modcast.fetch import ArcticShiftClient
    from modcast.retrieval import TfidfRetriever
    from modcast.subrules import rules_digest

    sub = sub.strip().removeprefix("r/")
    now = int(_t.time())
    start, end = now - days * 86400, now - 48 * 3600
    typer.echo(f"[1/4] fetching r/{sub} ({days} days of history, this can take a few minutes)…")
    client = ArcticShiftClient()
    try:
        chunk, total = 15 * 86400, 0
        cur = start
        while cur < end:
            total += client.fetch_window(sub, cur, min(cur + chunk, end))
            cur += chunk
            typer.echo(f"    …{total} posts so far")
    finally:
        client.close()
    typer.echo(f"[2/4] ingesting…")
    store = Store()  # the one writer; closed before anything else reads
    counts = store.ingest_jsonl_gz(config.RAW_DIR / f"{sub}.jsonl.gz")
    store.close()
    typer.echo(f"    {counts}")
    typer.echo("[3/4] building retrieval index…")
    ro = Store(read_only=True)
    df = ro.query(
        "SELECT id, title || chr(10) || chr(10) || selftext AS text, created_utc, label"
        " FROM posts WHERE subreddit = ? AND label IN ('survived','removed_mod') AND text_available"
        " ORDER BY id",
        [sub],
    ).df()
    n_removed = int((df["label"] == "removed_mod").sum())
    if len(df) < 100 or n_removed < 20:
        typer.secho(f"    warning: thin corpus ({len(df)} usable posts, {n_removed} removals) — "
                    f"forecasts will be weak; consider --days {days * 2}", fg="yellow")
    TfidfRetriever().fit(df).save(config.DATA_DIR / "index" / f"{sub}.joblib")
    ro.close()
    typer.echo("[4/4] caching published rules…")
    rules_digest(sub)
    rate = n_removed / len(df) if len(df) else 0
    typer.secho(f"r/{sub} onboarded: {len(df)} posts indexed, removal rate {rate:.0%}. "
                f"Note: accuracy on onboarded subs is not covered by the published eval.", fg="green")


def _predict_or_offer_onboard(**kwargs) -> None:
    """Interactive helper: on 'not in corpus', offer to onboard and retry once."""
    import re as _re

    try:
        predict(**kwargs)
    except typer.BadParameter as e:
        m = _re.search(r"r/(\w+) is not in ModCast's corpus", str(e))
        if not m:
            raise
        sub = m.group(1)
        if typer.confirm(f"r/{sub} isn't onboarded yet. Fetch its history and onboard now (~2-5 min)?",
                         default=True):
            onboard(sub=sub, days=90)
            predict(**kwargs)
        else:
            typer.echo("Skipped.")


@app.command()
def ingest() -> None:
    """Load every cached raw file into duckdb (idempotent upsert)."""
    store = Store()
    for path in sorted(config.RAW_DIR.glob("*.jsonl.gz")):
        counts = store.ingest_jsonl_gz(path)
        typer.echo(f"{path.name}: {counts}")
    typer.echo(f"totals: {store.counts()}")


@app.command()
def index(dense: bool = typer.Option(False, help="Also build the dense-embedding index (needs [embed] extra)")) -> None:
    """Build one retriever per subreddit over the index window (TF-IDF; --dense adds embeddings)."""
    from modcast.retrieval import DenseRetriever, TfidfRetriever
    from modcast.stats import _iso_to_epoch  # shared epoch conversion

    store = Store(read_only=True)
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
        if dense:
            dpath = DenseRetriever().fit(df).save(config.DATA_DIR / "index" / f"{sub}.dense.npz")
            typer.echo(f"{sub}: dense index -> {dpath}")


@app.command()
def induce(
    sub: list[str] = typer.Option(None, help="Subreddits (default: all configured)"),
    model: str = typer.Option(None),
    effort: str = typer.Option(None, help="Reasoning effort; default from .env LLM_MODEL"),
) -> None:
    """Induce a verified rulebook per subreddit (writes rulebooks/{sub}.md)."""
    from modcast.induce import induce as run_induce
    from modcast.llm import new_run_id

    store = Store(read_only=True)
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
    effort: str = typer.Option(None, help="Reasoning effort; default from .env LLM_MODEL"),
    rulebook: bool = typer.Option(False, help="Ablation: include induced rulebooks (measured harmful for forecasting)"),
) -> None:
    """Run the harness. Cheap predictors on the full test set; LLM predictors
    (and the cheap ones again, for same-case comparison) on the subsample."""
    from modcast import evaluate as E
    from modcast.baselines import BaseRatePredictor, LogisticPredictor
    from modcast.calibrate import CalibratedPredictor

    store = Store(read_only=True)
    train, test = E.build_split(store.con)
    typer.echo(f"train={len(train)} test={len(test)}")

    cheap = [BaseRatePredictor(), LogisticPredictor(), CalibratedPredictor(LogisticPredictor())]
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
                               use_rulebook=rulebook)
        preds = cheap + [OneShotPredictor(run_id=run_id, model=model), agent]
        out_dir = config.RESULTS_DIR / ("llm_rulebook" if rulebook else "llm")
        report = E.run_eval(preds, train, subsample, out_dir=out_dir)
        typer.echo(json.dumps({k: v["overall"] for k, v in report["predictors"].items()}, indent=1))
        spend = {"input_tokens": sum(f.input_tokens for f in agent.forecasts.values()),
                 "output_tokens": sum(f.output_tokens for f in agent.forecasts.values()),
                 "mean_turns": float(np.mean([f.turns for f in agent.forecasts.values()])) if agent.forecasts else None}
        (out_dir / "agent_spend.json").write_text(json.dumps(spend, indent=1))
        typer.echo(f"agent spend: {spend}")


@app.command()
def predict(
    sub: str = typer.Option(None, help="Subreddit (derived automatically with --url/--post-id)"),
    title: str = typer.Option(None),
    body: str = typer.Option("", help="Post body (or use --body-file)"),
    body_file: Path = typer.Option(None),
    flair: str = typer.Option(None, help="Link flair you will select when posting (e.g. 'Personal Project')"),
    post_id: str = typer.Option(None, help="Forecast a real archived post instead"),
    url: str = typer.Option(None, help="Reddit post URL — fetches title/body/subreddit automatically"),
    model: str = typer.Option(None),
    effort: str = typer.Option(None, help="Reasoning effort; default from .env LLM_MODEL"),
) -> None:
    """Preflight one post: probability + verified risk factors + fixes."""
    from modcast import agent as A
    from modcast import stats as S
    from modcast.llm import new_run_id
    from modcast.report import render
    from modcast.retrieval import load_retriever
    from modcast.schema import PostRecord, normalize
    from modcast.subrules import rules_digest

    store = Store(read_only=True)
    if url:
        from modcast.fetch import post_from_url

        record = normalize(post_from_url(url))
        sub = record.subreddit
        typer.echo(f"[fetched r/{sub} post {record.id}: {record.title!r}]")
    elif post_id:
        raw_row = store.query("SELECT * FROM posts WHERE id = ?", [post_id]).df()
        if raw_row.empty:
            from modcast.fetch import post_by_id

            typer.echo(f"[{post_id} not in local corpus — fetching from reddit/archive]")
            record = normalize(post_by_id(post_id))
        else:
            d = raw_row.iloc[0].to_dict()
            record = PostRecord(**{k: d[k] for k in PostRecord.__dataclass_fields__})
        sub = record.subreddit
        typer.echo(f"[r/{sub} post {record.id}: {record.title!r}]")
    else:
        if title is None or sub is None:
            raise typer.BadParameter("--sub and --title required (or use --url / --post-id)")
        if body_file:
            body = body_file.read_text()
        record = normalize({
            "id": f"draft-{int(time.time())}", "subreddit": sub,
            "created_utc": int(time.time()), "title": title, "selftext": body,
            "is_self": True, "link_flair_text": flair,
        })
    index_path = config.DATA_DIR / "index" / f"{sub}.joblib"
    if not index_path.exists():
        store.close()  # free the read handle so an in-process onboard can write
        known = ", ".join(config.SUBREDDITS)
        raise typer.BadParameter(
            f"r/{sub} is not in ModCast's corpus yet (available: {known}). "
            f"Onboard it first: modcast onboard {sub}"
        )
    rb_path = config.RULEBOOK_DIR / f"{sub}.md"
    ctx = A.AgentContext(
        con=store.con,
        retriever=load_retriever(sub),
        subreddit=sub,
        # curated subs use their regime-aware window; onboarded subs use all their data
        window=config.index_window(sub) if sub in config.SUBREDDITS else None,
    )
    run_id = new_run_id("predict")
    # final config: the induced rulebook is NOT fed to the forecaster (measured
    # to anchor forecasts toward the base rate and hurt accuracy — see changelog);
    # it remains a standalone artifact for humans and rule_ref citations.
    fc = A.forecast(
        ctx, record, run_id=run_id, rulebook="",
        published_rules=rules_digest(sub), model=model, effort=effort,
    )
    base = S.removal_rate(store.con, sub, window=ctx.window)["rate"]
    text = render(record, fc, base_rate=base)
    out = config.RESULTS_DIR / "reports" / f"{record.id}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    from modcast.report import colorize

    typer.echo(colorize(text))
    typer.echo(f"[saved {out} | trajectory {config.RESULTS_DIR / 'trajectories' / run_id}/]")


if __name__ == "__main__":
    app()
