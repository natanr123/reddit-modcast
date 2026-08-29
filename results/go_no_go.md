# Go/no-go: label quality probe (2026-08-29)

Before building anything we verified the load-bearing assumptions against the live
Arctic Shift API (sample: 100 posts per subreddit, window 2026-08-20 → 2026-08-21,
all posts old enough to have second-pass labels).

## Findings

1. **`_meta` labels are present via the API** (not only in the dumps):
   `was_deleted_later`, `removal_type`, `retrieved_2nd_on`, `is_edited` on 100/100 posts.
   Observed `removal_type` values: `moderator`, `deleted` (author), `reddit`, absent (survived).
2. **Removed posts keep their text.** In r/AmItheAsshole, 21/21 moderator-removed posts
   had full selftext (median 1,825 chars) captured before removal.
3. **First retrieval is near-instant**: median 24 s after posting (p90 ≈ 38 min).
   "Features at post time" is therefore honest for text captured at first pass.
4. **Fast removals are a separate population.** Posts removed *before* first snapshot
   appear with `removed_by_category` set and no body text. The label must union both
   signals; textless removals keep their label but are excluded from text-based eval
   (reported separately).

## Subreddit selection probe

| subreddit | n | self% | text% | mod | author-del | survived | removal rate* |
|---|---|---|---|---|---|---|---|
| relationship_advice | 100 | 100 | 26 | 64 | 21 | 15 | 81% |
| legaladvice | 100 | 100 | 93 | 16 | 23 | 61 | 21% |
| personalfinance | 100 | 92 | 66 | 47 | 14 | 37 | 56% |
| unpopularopinion | 100 | 100 | 54 | 58 | 20 | 21 | 74% |
| AskDocs | 100 | 64 | 90 | 9 | 19 | 72 | 11% |
| NoStupidQuestions | 100 | 100 | 59 | 27 | 24 | 48 | 37% |
| AskHistorians | 100 | 100 | 74 | 25 | 11 | 64 | 28% |
| TrueOffMyChest | 100 | 100 | 18 | 74 | 15 | 9 | 89% |
| AmItheAsshole | 100 | — | 85 | 26 | 24 | 50 | 34% |

\* mod+admin removals among decided posts (author-deletes excluded).

**Insight that shaped selection:** ultra-high-removal subs (r/TrueOffMyChest 89%,
r/relationship_advice 81%) remove so fast that most text is gone before the first
snapshot — unusable for text modeling. The sweet spot is strict-but-not-instant.

**Chosen:** AmItheAsshole (format rules), legaladvice (topic rules),
personalfinance (topic + automod), unpopularopinion (heavy-automod stress case).

## Decision

**GO.** All three go/no-go checks passed. Fallback (QueryDoctor) retired.

## Addendum (full-corpus ingest, 2026-08-29): two findings the probe missed

1. **Restored posts were mislabeled.** `_meta.was_initially_deleted=true` means
   a post was removed at the first snapshot but is AVAILABLE at the second —
   filter-first automod held it and a human mod approved it. The final state is
   *survived*; the initial label logic counted these as removals. Fixed:
   `was_initially_deleted` now takes precedence in `schema.label()`.
2. **r/AmItheAsshole changed moderation regimes in July 2026.** Jan–Jun, ~95%
   of new AITA posts were removed at creation (`removed_by_category` set within
   ~20 s of posting, most never restored) — filter-first moderation. From July
   the pattern collapses to ~35% conventional removals, matching our August
   probe. Monthly evidence (survived / removed / removed-at-snapshot-1):
   Jan 506/8484/5791 … Jun 250/9490/7089 → **Jul 2697/5247/2394, Aug 2457/3742/1036**.
   A forecaster trained on the January regime would be confidently wrong in
   August; a random train/test split would have hidden this entirely. AITA's
   index window therefore starts 2026-07-01 (`config.SUB_INDEX_START`).
