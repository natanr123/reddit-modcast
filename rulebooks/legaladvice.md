> NOTE: these rules are real and statistically verified, but the ablation (changelog #6)
> showed feeding them to the forecaster HURTS accuracy — they are a human-readable
> artifact and induction demo, deliberately not part of the shipped forecaster's context.

# Rulebook: r/legaladvice
Base removal rate 18.9% (n=47952), window 2026-01-01..2026-07-31.
Every entry below was verified by code against the labeled corpus.

- Post (title+body) never mentions a location/jurisdiction — automod requires a stated location for legal context.
  evidence: removal 99.5% (n=2195) when TRUE vs 15.0% (n=45757) when FALSE, effect +84.5%.
  predicate: `lower(selftext) NOT LIKE '%location%' AND lower(title) NOT LIKE '%location%'`
  maps to: Location requirement rule
- Selftext body is extremely short (under ~100-200 characters), indicating insufficient detail/effort for a legal question.
  evidence: removal 76.3% (n=3530) when TRUE vs 14.3% (n=44422) when FALSE, effect +62.1%.
  predicate: `length(selftext) < 200`
  maps to: Minimum detail/length requirement
- Post is flagged NSFW (over_18), which correlates with higher removal (explicit/inappropriate content).
  evidence: removal 46.6% (n=348) when TRUE vs 18.6% (n=47604) when FALSE, effect +27.9%.
  predicate: `over_18 = true`
  maps to: NSFW/content policy
- Post body contains a raw link (http/www), often indicating spam, self-promotion, or an off-site reference disallowed by mods.
  evidence: removal 35.9% (n=796) when TRUE vs 18.6% (n=47156) when FALSE, effect +17.4%.
  predicate: `selftext ILIKE '%http%' OR selftext ILIKE '%www.%'`
  maps to: No links/self-promotion rule

## Observations
The dominant driver by far is whether the poster states a location anywhere in the title/body — missing location correlates with ~99% removal vs ~15% baseline, and is largely redundant with (heavily overlapping) short-post-length, which alone also shows a huge lift (76-93% removal for <200/<100 char posts). Links and NSFW flags show moderate, weaker lifts (~2x) and smaller sample sizes. Several plausible hypotheses turned out NOT to matter: titles containing 'sue' were actually removed slightly *less* than average (lift 0.84), self-promotional phrasing ('I built/created a tool') showed only a negligible lift (~1.2), non-self (link) posts were too rare to matter, author_flair_text was essentially always null (unusable signal), and speculative sentencing-prediction titles ('how much time', 'prison time') had too small a sample (n=52) to trust despite a moderate lift.
