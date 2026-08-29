"""Fetcher tests: pagination + manifest against a mocked transport, one live smoke."""
from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

import modcast.fetch as fetch
from modcast.fetch import ArcticShiftClient, iter_cached

ROOT = Path(__file__).resolve().parent.parent

T0 = 1_787_184_000  # 2026-08-20 00:00 UTC


def _post(i: int, cu: int) -> dict:
    return {"id": f"mock{i}", "subreddit": "mocksub", "created_utc": cu}


class FakeAPI:
    """Mock Arctic Shift: ascending /posts/search pages and /posts/ids."""

    def __init__(self, posts: list[dict], page_cap: int = 2):
        self.posts = sorted(posts, key=lambda p: p["created_utc"])
        self.page_cap = page_cap
        self.requests: list[httpx.Request] = []
        self.fail_statuses: list[int] = []  # consumed one per request

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail_statuses:
            status = self.fail_statuses.pop(0)
            headers = {"X-RateLimit-Reset": "0"} if status == 429 else {}
            return httpx.Response(status, headers=headers, json={"error": "nope"})
        p = request.url.params
        if request.url.path.endswith("/posts/ids"):
            ids = p["ids"].split(",")
            return httpx.Response(200, json={"data": [x for x in self.posts if x["id"] in ids]})
        after, before = int(p["after"]), int(p["before"])
        page = [x for x in self.posts if after < x["created_utc"] < before][: self.page_cap]
        return httpx.Response(200, json={"data": page})

    def client(self, raw_dir: Path) -> ArcticShiftClient:
        return ArcticShiftClient(
            raw_dir=raw_dir, page_sleep=0, transport=httpx.MockTransport(self.handler)
        )


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(fetch.time, "sleep", slept.append)
    return slept


def test_pagination_advances_after(tmp_path):
    api = FakeAPI([_post(i, T0 + 10 * (i + 1)) for i in range(5)], page_cap=2)
    posts = list(api.client(tmp_path).iter_window("mocksub", T0, T0 + 3600))
    assert [p["id"] for p in posts] == [f"mock{i}" for i in range(5)]
    # pages of 2,2,1 then the empty page that stops the loop
    assert len(api.requests) == 4
    afters = [int(r.url.params["after"]) for r in api.requests]
    assert afters == [T0, T0 + 20, T0 + 40, T0 + 50]


def test_fetch_window_cache_and_manifest(tmp_path):
    api = FakeAPI([_post(i, T0 + i + 1) for i in range(3)])
    client = api.client(tmp_path)
    assert client.fetch_window("mocksub", T0, T0 + 100) == 3
    manifest = json.loads((tmp_path / "mocksub.manifest.json").read_text())
    assert manifest["windows"] == [[T0, T0 + 100]]
    with gzip.open(tmp_path / "mocksub.jsonl.gz", "rt") as f:
        assert len(f.readlines()) == 3
    # re-run skips: no new requests, nothing appended
    n_requests = len(api.requests)
    assert client.fetch_window("mocksub", T0, T0 + 100) == 0
    assert len(api.requests) == n_requests
    # sub-window of a completed window is also covered
    assert client.window_done("mocksub", T0 + 10, T0 + 50)
    assert not client.window_done("mocksub", T0 + 50, T0 + 200)
    assert [p["id"] for p in iter_cached("mocksub", tmp_path)] == ["mock0", "mock1", "mock2"]


def test_iter_cached_dedupes(tmp_path):
    with gzip.open(tmp_path / "mocksub.jsonl.gz", "at") as f:
        for i in [0, 1, 0]:
            f.write(json.dumps(_post(i, T0 + i)) + "\n")
    assert [p["id"] for p in iter_cached("mocksub", tmp_path)] == ["mock0", "mock1"]


def test_429_and_5xx_retries(tmp_path, no_sleep):
    api = FakeAPI([_post(0, T0 + 1)])
    api.fail_statuses = [429, 500]
    page = api.client(tmp_path).search_page("mocksub", T0, T0 + 100)
    assert [p["id"] for p in page] == ["mock0"]
    assert len(api.requests) == 3
    assert no_sleep == [1.0, 2]  # X-RateLimit-Reset(0)+1, then 2**1 backoff


def test_fetch_by_ids_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "IDS_PER_REQUEST", 2)
    api = FakeAPI([_post(i, T0 + i) for i in range(5)])
    got = api.client(tmp_path).fetch_by_ids([f"mock{i}" for i in range(5)])
    assert [p["id"] for p in got] == [f"mock{i}" for i in range(5)]
    assert len(api.requests) == 3


def test_month_windows():
    spec = importlib.util.spec_from_file_location(
        "make_dataset", ROOT / "scripts" / "make_dataset.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    windows = mod.month_windows("2026-01-15", "2026-03-10")
    assert len(windows) == 3
    assert windows[0][1] == windows[1][0] and windows[1][1] == windows[2][0]  # contiguous
    assert windows[0][0] == mod._epoch(__import__("datetime").date(2026, 1, 15))
    # inclusive end: last window reaches the start of 2026-03-11
    assert windows[-1][1] == mod._epoch(__import__("datetime").date(2026, 3, 11))


@pytest.mark.network
def test_live_smoke():
    """One real request: an hour of r/legaladvice on 2026-08-20."""
    client = ArcticShiftClient(page_sleep=0)
    try:
        posts = client.search_page("legaladvice", T0, T0 + 3600)
    finally:
        client.close()
    assert posts, "expected at least one post in the hour"
    assert all(T0 < p["created_utc"] < T0 + 3600 for p in posts)
    assert all("id" in p and "subreddit" in p for p in posts)
