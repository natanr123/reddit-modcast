> NOTE: these rules are real and statistically verified, but the ablation (changelog #6)
> showed feeding them to the forecaster HURTS accuracy — they are a human-readable
> artifact and induction demo, deliberately not part of the shipped forecaster's context.

# Rulebook: r/unpopularopinion
Base removal rate 87.6% (n=41393), window 2026-01-01..2026-07-31.
Every entry below was verified by code against the labeled corpus.

- Posts with a short body (selftext under ~300 characters) are removed at a much higher rate (96.6%) than longer posts (67.9%). This likely reflects a minimum-effort/length requirement (and includes link-only or nearly-empty text posts).
  evidence: removal 96.6% (n=28473) when TRUE vs 67.9% (n=12920) when FALSE, effect +28.7%.
  predicate: `length(selftext) < 300`
  maps to: Minimum post length / low-effort removal
- Posts lacking a link flair are removed at a near-certain rate (99.5%) vs 86.7% when flair is present, indicating flair is required/enforced by automod.
  evidence: removal 86.7% (n=38324) when TRUE vs 99.5% (n=3069) when FALSE, effect -12.9%.
  predicate: `link_flair_text IS NULL`
  maps to: Flair required
- Titles that explicitly include the phrase 'unpopular opinion' are removed more often (98.8% vs 87.4%), likely because the phrase is redundant/banned in the title.
  evidence: removal 99.6% (n=559) when TRUE vs 87.5% (n=40834) when FALSE, effect +12.2%.
  predicate: `title ILIKE '%unpopular opinion%'`
  maps to: Redundant phrase in title
- Titles phrased as questions (containing '?') are removed far more often (99.6% vs 87.4%), consistent with a rule that submissions must state an opinion, not ask a question.
  evidence: removal 98.8% (n=943) when TRUE vs 87.4% (n=40450) when FALSE, effect +11.5%.
  predicate: `title LIKE '%?%'`
  maps to: Must state an opinion, not a question
- NSFW-flagged (over_18) posts are removed somewhat more often (96.7% vs 87.5%).
  evidence: removal 96.7% (n=690) when TRUE vs 87.5% (n=40703) when FALSE, effect +9.2%.
  predicate: `over_18 = true`
  maps to: NSFW content restriction
- All-caps titles (shouting) show a higher removal rate (96.5% vs 87.6%), though sample size is small.
  evidence: removal 96.5% (n=86) when TRUE vs 87.6% (n=41307) when FALSE, effect +8.9%.
  predicate: `title = upper(title) AND length(title) > 5`
  maps to: Formatting / all-caps titles discouraged

## Observations
Post length is the dominant driver: very short or empty selftext correlates almost perfectly with removal, and this remains true across multiple thresholds (100, 300, 1000, 2000, 5000 chars), each showing progressively lower removal as length increases. Note that 'text_available = false' produces an almost identical signal to short/empty selftext (likely the same underlying phenomenon of removed posts having stripped text) and was excluded as a rule since it's a data artifact of removal rather than a content trait that causes removal. Being a link post (is_self=false) was too rare (n=4) to be meaningful. Political topic keywords (trump/biden/politic) actually showed slightly LOWER removal, so topic-based political content is not specifically targeted here.
