"""Fetch the configured subreddits from Arctic Shift, month by month.

Default run covers the index window [INDEX_START..INDEX_END] and the test
window [TEST_START..TEST_END]; completed months are skipped via the cache
manifest, so the script is safe to interrupt and re-run.
"""
from __future__ import annotations

import datetime as dt

import typer

from modcast import config
from modcast.fetch import ArcticShiftClient

app = typer.Typer(add_completion=False)


def _epoch(d: dt.date) -> int:
    return int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp())


def month_windows(start: str, end: str) -> list[tuple[int, int]]:
    """[after, before) epoch windows covering start..end (inclusive dates), split at month starts."""
    cur = dt.date.fromisoformat(start)
    hi = dt.date.fromisoformat(end) + dt.timedelta(days=1)
    windows: list[tuple[int, int]] = []
    while cur < hi:
        nxt = min((cur.replace(day=1) + dt.timedelta(days=32)).replace(day=1), hi)
        windows.append((_epoch(cur), _epoch(nxt)))
        cur = nxt
    return windows


@app.command()
def main(
    sub: list[str] = typer.Option(None, "--sub", help="Subreddit(s); default: config.SUBREDDITS"),
    start: str = typer.Option(None, help="Custom window start (YYYY-MM-DD); overrides the splits"),
    end: str = typer.Option(None, help="Custom window end (YYYY-MM-DD, inclusive)"),
    test_only: bool = typer.Option(False, "--test-only", help="Only the test window"),
    index_only: bool = typer.Option(False, "--index-only", help="Only the index window"),
) -> None:
    """Fetch Arctic Shift archives into data/raw/, resumable via manifests."""
    subs = sub or config.SUBREDDITS
    if start or end:
        ranges = [(start or config.INDEX_START, end or config.TEST_END)]
    else:
        ranges = []
        if not test_only:
            ranges.append((config.INDEX_START, config.INDEX_END))
        if not index_only:
            ranges.append((config.TEST_START, config.TEST_END))
    client = ArcticShiftClient()
    try:
        for s in subs:
            for r_start, r_end in ranges:
                for after, before in month_windows(r_start, r_end):
                    tag = dt.datetime.fromtimestamp(after, dt.timezone.utc).strftime("%Y-%m-%d")
                    if client.window_done(s, after, before):
                        typer.echo(f"r/{s} {tag}: cached, skipping")
                        continue
                    typer.echo(f"r/{s} {tag}: fetching...")
                    n = client.fetch_window(s, after, before)
                    typer.echo(f"r/{s} {tag}: {n} posts")
    finally:
        client.close()


if __name__ == "__main__":
    app()
