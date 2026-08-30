> NOTE: these rules are real and statistically verified, but the ablation (changelog #6)
> showed feeding them to the forecaster HURTS accuracy — they are a human-readable
> artifact and induction demo, deliberately not part of the shipped forecaster's context.

# Rulebook: r/personalfinance
Base removal rate 51.3% (n=56407), window 2026-01-01..2026-07-31.
Every entry below was verified by code against the labeled corpus.

- Post body (selftext) is too short (under ~200 characters) — lacks the detail needed for a personal finance question and gets removed as low-effort/insufficient info.
  evidence: removal 89.9% (n=27293) when TRUE vs 15.0% (n=29114) when FALSE, effect +74.9%.
  predicate: `length(selftext) < 200`
  maps to: Minimum length / provide details rule
- Post body contains no numbers/figures at all (no income, dollar amounts, percentages, etc.) — treated as too vague to be a real personal-finance question.
  evidence: removal 85.0% (n=29791) when TRUE vs 13.5% (n=26616) when FALSE, effect +71.5%.
  predicate: `NOT regexp_matches(selftext, '[0-9]')`
  maps to: Provide specific financial details rule
- Post has no link flair assigned — flair is required (and mods/automod appear to only apply flair to approved posts), so unflaired posts are removed at extremely high rate.
  evidence: removal 98.2% (n=10553) when TRUE vs 40.5% (n=45854) when FALSE, effect +57.8%.
  predicate: `link_flair_text IS NULL`
  maps to: Flair required rule
- All-caps titles are removed almost universally.
  evidence: removal 99.8% (n=400) when TRUE vs 50.9% (n=56007) when FALSE, effect +48.8%.
  predicate: `upper(title) = title AND length(title) > 10`
  maps to: No all-caps titles rule
- Posts mentioning tuition/scholarship funding are removed far less than average, contrary to the intuition that 'need money for tuition' begging posts get removed — most tuition-related posts here are legitimate detailed questions.
  evidence: removal 14.9% (n=510) when TRUE vs 51.6% (n=55897) when FALSE, effect -36.7%.
  predicate: `lower(title) LIKE '%tuition%' OR lower(title) LIKE '%scholarship%' OR lower(selftext) LIKE '%tuition%'`
- Non-self (link) posts are removed far more often than self/text posts — the sub expects a text submission with your situation, not a link.
  evidence: removal 81.1% (n=7526) when TRUE vs 46.7% (n=48881) when FALSE, effect +34.4%.
  predicate: `is_self = false`
  maps to: Self-post only rule
- Generic discussion/poll-style titles ('Do you...', 'Does anyone...', 'What would you...') are removed somewhat more than average as off-topic discussion rather than a personal finance situation.
  evidence: removal 80.0% (n=1598) when TRUE vs 50.4% (n=54809) when FALSE, effect +29.5%.
  predicate: `lower(title) LIKE '%do you%' OR lower(title) LIKE '%does anyone%' OR lower(title) LIKE '%what would you%'`
  maps to: No generic discussion/polls rule
- Posts explicitly discussing charitable giving/donations/fundraising amounts (as a personal budgeting decision) are actually removed LESS than average — they are legitimate PF topics, not off-topic begging as might be assumed.
  evidence: removal 30.8% (n=130) when TRUE vs 51.3% (n=56277) when FALSE, effect -20.6%.
  predicate: `lower(title) LIKE '%charity%' OR lower(selftext) LIKE '%charity%' OR lower(selftext) LIKE '%donat%' OR lower(selftext) LIKE '%fundrais%'`
- Posts asking for app/tool recommendations are removed somewhat more than average (treated as recommendation-request spam rather than personal situation).
  evidence: removal 66.1% (n=2642) when TRUE vs 50.5% (n=53765) when FALSE, effect +15.6%.
  predicate: `lower(title) LIKE '%app%' OR lower(title) LIKE '%recommend%'`
  maps to: No product recommendation requests rule

## Observations
Side-hustle/'make money' phrasing and NSFW flag showed only weak or noisy signal (small samples) and were dropped. Surprisingly, topics that sound like begging (charity/donations, tuition funding) actually had LOWER removal rates than baseline — these appear to be legitimate, detailed personal-finance questions rather than off-topic pleas, contradicting the naive reading of the removed examples. The dominant, highly reliable drivers are structural/format based: missing flair, missing selftext length/detail, lack of any numeric figures, non-self (link) posts, and all-caps titles — all showing 2x-6x lift, consistent with automod enforcing format and detail requirements rather than topic-based content rules.
