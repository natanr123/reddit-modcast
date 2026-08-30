# ModCast — a weather forecast for your Reddit post

**Input:** a subreddit + a draft post. **Output:** the probability moderation removes it
within 36 hours — with machine-verified risk factors, cited precedent posts, and concrete fixes.

Scored the way real weather forecasts are scored: with the **Brier score**, against **real
moderation outcomes**. 🎬 **Solution video (5 min):** https://www.youtube.com/watch?v=PLScEY_HQO0

```
$ modcast predict --sub personalfinance --title "Need money advice ASAP" --body "..."
  ⋯ evidence dossier: base rate 51%, 25 similar posts retrieved (8% of them removed)
  ⋯ testing: do bodies under 200 characters get removed more here?
  ⋯ reading precedent 1thit3q
  ⋯ verifying cited evidence…
**Removal risk: 98% — SEVERE WEATHER** (subreddit base rate: 51%)
```

## Who has this problem

Anyone posting in a strictly moderated subreddit. In our measured corpora, **21%
(r/legaladvice) to 74% (r/unpopularopinion) of genuine posts are removed by moderation** —
usually for tripping format, flair, length, or topic rules the poster never read. The poster
loses the post and the answers they needed (I hit this myself posting my own side project —
that run is in the demo). Today's "solution" is pasting the draft into a chatbot; we measured
that practice, and it performs **statistically the same as a four-entry lookup table**.

## Why an agent (the measured answer)

The gap is not the model and not information access — it is the **investigation loop**. Same
LLM everywhere:

| Method (same 600 held-out posts, same LLM where used) | Brier ↓ | AUC ↑ | ECE ↓ |
|---|---|---|---|
| Subreddit base rate only (no model) | 0.1899 | 0.735 | 0.048 |
| Logistic regression, 20 features | 0.1915 | 0.820 | 0.159 |
| Calibrated logistic (strongest classical baseline) | 0.1636 | 0.821 | 0.069 |
| One-shot LLM (today's practice) | 0.1845 | 0.782 | 0.068 |
| One-shot LLM **+ rules + base rate in the prompt** | 0.1887 | 0.821 | 0.147 |
| **ModCast agent (shipped config)** | **0.1277** | **0.889** | 0.061 |

Every agent-vs-baseline gap is significant at **p < 10⁻⁴** (paired bootstrap, 10,000
resamples, identical posts). Handing the LLM the context as text bought *nothing* — the
informed one-shot ties the uninformed one (p = 0.35) and its calibration worsens. Giving the
same model **tools to interrogate the corpus** cuts error by 31%.

How the agent earns that: per forecast it (1) receives a code-computed evidence dossier
(base rate with CI, 25 nearest labeled precedents, deterministic features), (2) invents and
tests hypotheses about *this* post as real corpus queries (a validated SQL mini-language —
the model never eyeballs statistics), (3) reads the closest precedents in full (measured:
4.0 `read_post` calls per forecast, 100% of forecasts), and (4) must deliver through a strict
schema whose every risk factor is **independently re-verified** — cited precedents are
checked against the database, statistical claims are re-executed, and unsupported factors
are bounced back for one repair round, then dropped.

- **Real labels, zero hand-annotation:** the [Arctic Shift](https://github.com/ArthurHeitmann/arctic_shift)
  archive snapshots every post ~20 s after creation and again 36 h later; the diff between
  snapshots — for 269k posts across 4 subreddits — *is* the ground truth.
- **Leak-proof by construction:** train/index on Jan–Jul 2026, test on Aug 1–25; per-query
  temporal masking; at-post-time features only; per-subreddit regime windows (see changelog #2).
- **Stable:** re-forecasting the same posts 5× moves the median answer by **±1.6 points**.
- **Cross-engine:** the architecture replicates on a second LLM (claude-sonnet-5, n=100:
  agent 0.155 vs one-shot 0.249 — where naive prompting was *worse than knowing nothing*).

## Improvement Changelog

Every entry was measured on identical seeded eval posts; deltas cite committed evidence.

| # | What we tried and why | Evidence | Decision / learning |
|---|---|---|---|
| 0 | Go/no-go: verify archive labels, text capture, retrieval delay; pick subreddits | `results/go_no_go.md` | GO. Ultra-fast-removal subs lose text before the first snapshot → label must union `removed_by_category` + `_meta.removal_type`; chose 4 strict-but-not-instant subs |
| 1 | Full-corpus ingest exposed two label traps | `results/go_no_go.md` (addendum) | Restored posts (`was_initially_deleted`) are survivors, not removals; **r/AmItheAsshole switched moderation regimes in July 2026** → per-sub index windows. A random split would have hidden both |
| 2 | Classical baselines (base-rate, logistic, +isotonic) | `results/eval_latest.md` | Discrimination is learnable (AUC 0.82) but raw logistic is badly miscalibrated — Brier-worse than the base rate. Calibration is half the game |
| 3 | Rulebook induction: model proposes removal drivers, code re-verifies each on the corpus, only confirmed rules kept | `rulebooks/*.md` | 23 verified rules (e.g. unflaired r/personalfinance posts: 98.2% removed, n=10,553), incl. counterintuitive keeps. Cost $0.39 |
| 4 | **Agent v1**: dossier + query/read tools + citation verifier + rulebook memory | `results/llm_v1/` | Brier 0.1525 — beats every baseline, but AITA barely discriminates: predictions hug the base rate |
| 5 | Diagnose: rulebook = anchoring. Try an investigation-mandatory prompt (kept rulebook) | `results/llm_v2/` | **No effect** (0.1527). You cannot instruct a model to unsee context |
| 6 | **Remove the rulebook from the forecaster's context** | `results/llm_norulebook_v1/` | **0.1246** — the project's biggest jump; AITA Brier 0.20 → 0.09. Verified memory measurably *hurt* |
| 7 | Replicate #6 with a fresh cache (best-of-grid suspicion) | `results/llm_norulebook/` | 0.1257 — replicated; not selection luck. Full 2×2 prompt×rulebook grid archived |
| 8 | **Shipped config**: + regime-aware base-rate anchor + statistically-verifiable risk factors | `results/llm/eval_latest.md` | **0.1277 / AUC 0.889**; every baseline beaten at p<10⁻⁴; AITA becomes the *best* discriminated sub (AUC 0.927) |
| 9 | Dense-embedding retrieval (the "vector DB" upgrade) | `results/llm_dense/` | Statistical tie with TF-IDF (0.1305 vs 0.1277, p=0.27). Kept TF-IDF: equal accuracy, zero heavy deps. The vector DB would have been résumé-driven engineering |
| 10 | Informed one-shot: rules + base rate as prompt context, no tools | `results/informed_oneshot.json` | 0.1887 — context alone buys nothing and calibration worsens (ECE 0.068→0.147). Second independent observation of context-anchoring |
| 11 | Run-to-run stability, 20 posts × 5 fresh investigations | `results/stability.json` | Median std ±1.6pp. Residual instability concentrates in AITA — where moderation itself is most discretionary |

**Main failure mode:** *verified memory made the agent worse.* The induced rulebooks — the
component agent frameworks tell you to add, statistically verified and true — anchored
forecasts toward subreddit averages and crowded out post-specific investigation. A sterner
prompt didn't fix it (#5); only removing the content did (#6); and the informed one-shot
(#10) reproduced the same anchoring in a second setting.

**Hot take:** *every component must pay rent in Brier score.* We built the textbook agent —
tools, verifier, memory, then embeddings — and measurement kept only tools + verifier. The
two components everyone assumes help (memory, dense retrieval) measurably didn't. An agent
architecture is a set of hypotheses, not a checklist; the teams that ablate will quietly
outperform the teams that accumulate.

## Claims and evidence

| Claim | Strength | Where |
|---|---|---|
| Agent beats one-shot practice, both engines | p<10⁻⁴ (n=600); p=0.003 (n=100, sonnet) | `results/llm/`, changelog #8 |
| Agent beats the strongest classical baseline | p<10⁻⁴ (+0.036 Brier) | `results/llm/eval_latest.md` |
| Context-in-prompt ≠ investigation | informed one-shot ties naive (p=0.35) | `results/informed_oneshot.json` |
| Rulebook memory harms forecasting | p<10⁻⁴, replicated | `results/llm_v1..norulebook/` |
| Dense retrieval ≈ TF-IDF at this scale | tie, p=0.27 | `results/llm_dense/` |
| Forecasts are stable, not sampling noise | median ±1.6pp over 5 reps | `results/stability.json` |

## Reproduction (from a clean machine)

> Full step-by-step guide with expected outputs, per-claim reproduction commands, runtime
> and cost tables: **[REPRODUCTION.md](REPRODUCTION.md)**. The short version:

```bash
git clone <this repo> && cd reddit-modcast
python -m venv .venv && .venv/bin/pip install -e ".[dev]"   # Python ≥3.11 (built on 3.14)
#   Debian/Ubuntu: if venv creation fails, `sudo apt install python3-venv` first
.venv/bin/pytest -q                     # 40 unit tests, no network, no keys

# data: fetched from the public Arctic Shift API (never redistributed here)
.venv/bin/python scripts/make_dataset.py     # ~250k posts, ~45 min, resumable
.venv/bin/modcast ingest                     # ~20 s
.venv/bin/modcast index                      # ~2 min

.venv/bin/modcast eval                       # baselines on 1,000 test posts, free, ~3 min

export ANTHROPIC_API_KEY=sk-ant-...          # LLM stages (or copy .env.example → .env)
.venv/bin/modcast eval --with-llm --llm-posts 25   # ~100 posts ≈ $8 on claude-sonnet-5
.venv/bin/modcast predict --url "https://www.reddit.com/r/personalfinance/comments/..."
.venv/bin/modcast                            # or just the interactive wizard
```

- **Zero-setup demo path:** `modcast` on a fresh install self-bootstraps — name any
  subreddit and it offers to onboard it (fetch → label → index, ~3 min).
- **Determinism:** all sampling is seeded (`RANDOM_SEED` in `modcast/config.py`); exact dep
  versions in `requirements.lock.txt`; the corpus refetches reproducibly (we deleted ours
  and refetched: counts matched within ±2 posts per subreddit).
- **Cost/runtime:** a single forecast on the judge default (claude-opus-5, high effort)
  measures **~$0.14 and ~60 s** (verified from a clean clone); ≈ $0.08/post at
  claude-sonnet-5. Our reported n=600 runs used the flat-rate `codex` CLI backend
  (`LLM_MODEL=codex-cli gpt-5.6-sol medium` in `.env`) — both backends drive identical
  code paths.
- Optional extra: `pip install -e ".[embed]"` + `MODCAST_RETRIEVER=dense` reproduces the
  dense-retrieval ablation (torch; not needed otherwise).

Expected outputs: the tables in `results/` regenerate into `tmp/generated/results/` (set
`MODCAST_OUT_DIR=` empty to write into `results/` directly). Agent trajectories for every
forecast appear under `.../results/trajectories/` — curated examples with a reading guide
are committed in `results/trajectories/`.

## Limitations (known and stated)

- **36-hour label window**: "survives" means survived 36 h (where most removals happen);
  later removals count as survivals.
- **Posts removed within ~20 s** leave no text in the archive; they keep their label but are
  excluded from text-based eval — so real-world draft risk is, if anything, *underestimated*.
- **Flair causality**: flair absence predicts removal partly because mods flair approved
  posts. Selecting flair at submit time is still a real lever (many automods hard-remove
  unflaired posts), but the 98% figure mixes both mechanisms.
- **Scope of measured accuracy**: 4 subreddits, one test month, one platform. Moderation
  drifts (we caught one regime change mid-project); numbers describe Aug 2026. Onboarded
  subreddits work but are outside the measured eval.
- The paired bootstrap treats posts as independent; posts within a subreddit share an
  automod regime (4 clusters is too few for a clustered bootstrap, so we disclose instead).

## Ethics & ground rules

ModCast is a **compliance assistant**: it helps posters follow rules subreddits publish, and
when a draft violates rules outright the correct output is "don't post this here" — not an
evasion rewrite. It is read-only by design (never posts or acts on Reddit); the human
decides. Only public archive data is used, fetched at run time rather than redistributed;
no author identities are stored or used as features; no credentials ship in the repo.

**What existed before the competition / what we added:** everything in this repository was
built during the event. External components used as-is: public Python libraries (see
`requirements.lock.txt`), the public Arctic Shift archive/API (credited above), and Reddit's
public post JSON. The Codex CLI wrapper concept was adapted from the author's own prior
utility code and generalized here.

## Coding-agent disclosure

Built with **Claude Code (Fable 5)** — including parallel subagent workflows for module
construction and detached overnight evaluation runs — with the **Codex CLI** (`gpt-5.6-sol`)
as a flat-rate inference backend for test iterations. Coding-agent use is required by this
challenge and fully embraced: the solution's own agent trajectories (every LLM call, tool
invocation, and verification round) are logged as JSONL — see **[TRAJECTORIES.md](TRAJECTORIES.md)**
for the per-agent map and an annotated walkthrough, with curated files in `results/trajectories/`.
