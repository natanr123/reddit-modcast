> NOTE: these rules are real and statistically verified, but the ablation (changelog #6)
> showed feeding them to the forecaster HURTS accuracy — they are a human-readable
> artifact and induction demo, deliberately not part of the shipped forecaster's context.

# Rulebook: r/AmItheAsshole
Base removal rate 62.3% (n=7942), window 2026-07-01..2026-07-31.
Every entry below was verified by code against the labeled corpus.

- Post is flagged NSFW (over_18)
  evidence: removal 97.1% (n=70) when TRUE vs 62.0% (n=7872) when FALSE, effect +35.2%.
  predicate: `over_18 = true`
  maps to: NSFW content rule
- Title contains 'update' (update posts face stricter/different removal criteria)
  evidence: removal 89.3% (n=75) when TRUE vs 62.0% (n=7867) when FALSE, effect +27.3%.
  predicate: `lower(title) LIKE '%update%'`
  maps to: Update post rule
- Title does not contain 'AITA' or 'WIBTA' (missing the required judgment-request tag)
  evidence: removal 83.3% (n=1525) when TRUE vs 57.3% (n=6417) when FALSE, effect +26.0%.
  predicate: `NOT (lower(title) LIKE '%aita%' OR lower(title) LIKE '%wibta%')`
  maps to: Title format rule requiring AITA/WIBTA tag
- All-caps titles are removed somewhat more often (shouting/formatting)
  evidence: removal 83.3% (n=36) when TRUE vs 62.2% (n=7906) when FALSE, effect +21.2%.
  predicate: `title = upper(title)`

## Observations
The strongest apparent predictors (link_flair_text IS NULL, text_available=false, selftext length <200-500 chars) turned out to be label-leakage artifacts: removed posts have their selftext replaced with a placeholder/empty text and never receive a verdict flair, so these variables essentially just re-encode the removal outcome itself rather than reflecting a real pre-removal posting rule. Genuine, actionable signals were much weaker: missing the AITA/WIBTA tag in the title, NSFW flag, and 'AITAH'/'update' title variants showed real but modest lifts (1.3-1.5x). Time-of-day, mention of spouse, URLs in body, and young-age mentions in text showed no meaningful or reliable effect. WIBTA-tagged posts were actually removed less than AITA-tagged ones, contrary to expectation.
