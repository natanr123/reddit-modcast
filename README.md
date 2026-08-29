# ModCast — a weather forecast for your Reddit post

**Input:** a subreddit + a draft post. **Output:** the probability your post
will be removed by moderation within 36 hours, with machine-verified risk
factors, cited precedent posts, and concrete fixes.

Scored the way real weather forecasts are scored: with the Brier score,
against real moderation outcomes.

## Who has this problem

Anyone posting in a strictly moderated subreddit. In our measured corpora,
21% (r/legaladvice) to 74% (r/unpopularopinion) of genuine posts get removed
by mods or automod — usually for tripping format, flair, length, or topic
rules the poster never read. The poster loses the post and the answers they
needed; mods burn hours removing posts that a pre-flight check would have
fixed. The current "solution" is pasting your draft into a chatbot and asking
"will this get removed?" — which produces a confident, uncalibrated guess.

## Why an agent (and not one prompt)

A one-shot LLM cannot know how r/personalfinance's automod actually behaves
this year. ModCast's agent can, because it investigates a real labeled
corpus before answering:

- **Real labels.** The [Arctic Shift](https://github.com/ArthurHeitmann/arctic_shift)
  archive captures every post minutes after creation and re-checks it 36 h
  later, recording whether moderation removed it. No synthetic data anywhere.
- **Verified statistics as tools.** The model never eyeballs data. It asks
  `compare_removal_rate("length(selftext) < 300")` and code answers with real
  rates and confidence intervals. (That example is real: 68% vs 12% removal
  in r/legaladvice.)
- **Induced rulebooks as memory.** Offline, the agent proposes removal-driver
  hypotheses per subreddit; every hypothesis is re-run by code and only
  statistically confirmed rules become the rulebook the forecaster reads.
- **Citation verification.** Every risk factor in a forecast must cite
  precedent post ids; a verifier independently checks each citation against
  the database and rejects unsupported claims.
- **Calibration, measured.** Probabilities are scored with Brier/AUC/ECE on a
  strictly later, held-out test window.

## Evaluation design (leak-proof by construction)

- **Task:** P(removed by moderation within 36 h) among decided posts;
  author self-deletes are excluded (not a moderation outcome).
- **Temporal split:** retrieval index, rulebooks, and all fitting use
  2026-01-01 → 2026-07-31; the test set is sampled from 2026-08-01 → 2026-08-25.
  Retrieval enforces a per-query temporal mask.
- **At-post-time features only:** second-pass fields (score, comments, ...)
  are barred from features; posts removed before first snapshot keep their
  label but are excluded from text-based eval (reported separately).
- **Baselines (same cases, same metric):**
  1. `base_rate` — predict the subreddit's training-window removal rate;
  2. `logistic` — 20 deterministic features, balanced logistic regression;
  3. `llm_oneshot` — the same LLM, one direct prompt, no tools (what people
     actually do today).
- **Primary metric:** Brier score (calibration-sensitive), plus AUC, ECE,
  and reliability curves.

## Reproduction

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q                      # unit tests, no network needed
.venv/bin/python scripts/make_dataset.py # fetch corpus from Arctic Shift API
.venv/bin/python -m modcast.cli ingest
.venv/bin/python -m modcast.cli index
export ANTHROPIC_API_KEY=...             # for LLM stages
.venv/bin/python -m modcast.cli induce   # build verified rulebooks
.venv/bin/python -m modcast.cli eval --with-llm
.venv/bin/python -m modcast.cli predict --sub legaladvice --title "..." --body "..."
```

Raw Reddit data is fetched, never redistributed; `data/` is gitignored.
Agent trajectories for every LLM run land in `results/trajectories/`.

## Improvement Changelog

Every iteration is measured on the same seeded eval cases; deltas cite the
committed results files.

| Stage | What we tried and why | Evidence | Decision / learning |
|---|---|---|---|
| 0 | Go/no-go: verify Arctic Shift labels, text capture, retrieval delay; pick subreddits | `results/go_no_go.md` | GO. Fast-removal subs lose text before first snapshot → label must union `removed_by_category` + `_meta.removal_type`; chose 4 strict-but-not-instant subs |
| 1 | Baselines (base-rate, logistic) on full test set | _pending_ | _pending_ |
| 2 | One-shot LLM baseline vs agent v1 (dossier + tools) | _pending_ | _pending_ |
| 3 | + induced rulebooks (ablation: `--no-rulebook`) | _pending_ | _pending_ |
| 4 | + citation verifier repair round | _pending_ | _pending_ |
| 5 | + calibration | _pending_ | _pending_ |

## Ethics

ModCast is a compliance assistant, not an evasion tool: it helps posters
follow rules that subreddits publish, mirrors what pre-post guides already
do, and its suggested fixes must keep the post honest. When a draft violates
a subreddit's rules outright, the correct output is "don't post this here" —
not a rewrite that sneaks it past automod. Only public archive data is used,
fetched at judge time rather than redistributed; no author identities are
used as features.

## Coding-agent disclosure

Built with Claude Code (Fable 5) using parallel subagent workflows for
module construction and this session's own agent for the LLM core; full
agent trajectories for every ModCast run ship in `results/trajectories/`.
