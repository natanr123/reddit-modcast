# Reproduction guide

Written for someone starting from a clean machine. Every command below was executed from a
fresh `git clone` into a fresh virtual environment before submission; the measured runtimes
and costs come from those runs.

## 0. Prerequisites

- Linux or macOS, **Python ≥ 3.11** (built and tested on 3.14), git, ~3 GB free disk
- Internet access (data is fetched from a public archive, never redistributed in this repo)
- For the LLM stages only: an Anthropic API key
- Debian/Ubuntu: `sudo apt install python3-venv` if venv creation fails

## 1. Setup (~2 min)

```bash
git clone https://github.com/natanr123/reddit-modcast.git
cd reddit-modcast
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q          # expected: 40 passed — no network, no keys needed
```

Exact dependency versions we ran with: `requirements.lock.txt` (install with
`.venv/bin/pip install -r requirements.lock.txt` if you want bit-identical libs).

## 2. Data (~45 min, unattended, resumable)

Source: the public [Arctic Shift](https://github.com/ArthurHeitmann/arctic_shift) Reddit
archive — it snapshots every post ~20 s after creation and again 36 h later; the diff is our
removal ground truth. The fetcher is polite (1 req/s, backoff on 429/5xx/422) and resumable
(re-running skips completed windows).

```bash
.venv/bin/python scripts/make_dataset.py   # 4 subreddits, Jan–Aug 2026, ~250k posts, ~140 MB
.venv/bin/modcast ingest                   # ~20 s → data/modcast.duckdb
.venv/bin/modcast index                    # ~2 min → TF-IDF retrievers per subreddit
```

Expected ingest totals (±2 posts per subreddit — the live archive honors deletion requests;
we verified this drift bound by deleting our corpus and refetching):

| subreddit | posts | removed (mod) | survived |
|---|---|---|---|
| AmItheAsshole | ~77.6k | ~54.4k | ~15.1k |
| legaladvice | ~68.3k | ~9.8k | ~42.5k |
| personalfinance | ~68.5k | ~31.0k | ~29.9k |
| unpopularopinion | ~54.7k | ~40.3k | ~6.0k |

## 3. Baselines — free, no keys (~3 min)

```bash
.venv/bin/modcast eval
```

Runs base-rate, logistic, and calibrated-logistic on 1,000 seeded test posts. Expected
output ≈ the committed `results/eval_latest.md` (deterministic given the corpus; the ±2-post
archive drift can shift the seeded sample slightly).

## 4. LLM stages — the agent itself

```bash
cp .env.example .env    # put your ANTHROPIC_API_KEY in it (never committed; .env is gitignored)
```

With only the key set, defaults are the **judge configuration**: Anthropic backend,
`claude-opus-5`, high effort, shipped agent config (no rulebook in context, TF-IDF
retrieval, anchor-first prompt).

**One forecast (~60 s, ~$0.14 — both measured):**

```bash
.venv/bin/modcast predict --sub legaladvice \
  --title "Landlord kept my deposit, what can I do?" \
  --body "My landlord in Ohio is keeping my 1200 dollar deposit for cleaning but the apartment was spotless. He never sent an itemized list. What are my options?"

.venv/bin/modcast predict --url "https://www.reddit.com/r/personalfinance/comments/1vwoib3/how_do_i_actually_invest_my_money/"
.venv/bin/modcast            # or the interactive wizard (self-bootstraps; can onboard ANY subreddit in ~3 min)
```

Each run prints the preflight report and writes the full agent trajectory (every tool call
and verification round) under `results/trajectories/`.

**Small evaluation (100 posts, ≈ $8 at claude-sonnet-5 / ≈ $14 at opus):**

```bash
.venv/bin/modcast eval --with-llm --llm-posts 25 --model claude-sonnet-5
```

Fits, scores, plots, and writes `.../results/llm/eval_latest.{json,md}` including per-post
predictions and paired-bootstrap significance for every method pair. Per-post forecasts are
cached (`results/pred_cache/`), so interrupted runs resume.

## 5. Reproducing each reported number

| Claim / table | Command | Committed evidence |
|---|---|---|
| Headline eval (agent vs 5 baselines, n=600) | `modcast eval --with-llm --llm-posts 150` | `results/llm/eval_latest.md` |
| Rulebook-ON ablation | same + `--rulebook` | `results/llm_v1/`, `results/llm_v2/` |
| Rulebook-OFF original + replication | archived runs | `results/llm_norulebook_v1/`, `results/llm_norulebook/` |
| Dense-retrieval ablation | `pip install -e ".[embed]"`, `modcast index --dense`, then eval with `MODCAST_RETRIEVER=dense` | `results/llm_dense/` |
| Informed one-shot (context-only rung) | `.venv/bin/python scripts/informed_oneshot.py` | `results/informed_oneshot.json` |
| Run-to-run stability (20×5) | `.venv/bin/python scripts/stability.py` | `results/stability.json` |
| Rulebook induction (verified memory) | `modcast induce` (~$0.40 at sonnet) | `rulebooks/*.md` |
| Label/regime findings | see methodology notes | `results/go_no_go.md` |

Notes: all sampling is seeded (`RANDOM_SEED` in `modcast/config.py`). Our n=600 headline
runs used the flat-rate Codex CLI backend (`LLM_MODEL=codex-cli gpt-5.6-sol medium` in
`.env`, requires a logged-in `codex` CLI); the Anthropic backend drives identical code paths
— the n=100 claude-sonnet-5 replication in the README used it. Expect individual forecasts
to vary run-to-run by ±1.6 points median (measured, `results/stability.json`); aggregate
metrics over ≥100 posts are stable to ~±0.003 Brier.

## 6. Runtime & cost summary

| Step | Time | Cost |
|---|---|---|
| install + tests | ~2 min | — |
| data fetch → index | ~50 min | free |
| baseline eval (n=1000) | ~3 min | free |
| one agent forecast | ~60 s | ~$0.14 (opus) / ~$0.08 (sonnet) |
| 100-post LLM eval | ~30 min | ~$8 (sonnet) |
| full 600-post arm | ~45 min | free on codex-cli / ~$48 (sonnet) |

## 7. Troubleshooting

- **`Cannot open database … read-only`/ lock errors** — another modcast process holds the
  duckdb writer (only `ingest`/`onboard` write); wait for it or close it. All read commands
  co-exist freely.
- **Transient 422/429/5xx from the archive** — handled automatically with backoff; a crashed
  fetch is resumable by re-running the same command.
- **`r/<sub> is not in ModCast's corpus`** — run `modcast onboard <sub>` (or accept the
  wizard's offer); ~3 min for a mid-size subreddit.
- **Empty `data/`** — everything self-creates on first touch; the wizard works from zero.
