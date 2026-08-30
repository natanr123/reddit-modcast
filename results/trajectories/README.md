# Agent trajectories — reading guide

Every LLM interaction in ModCast is logged as it happens, one JSONL file per conversation.
These are curated representative examples (the full set — 5,000+ files across all
evaluation runs — regenerates with any eval command; complete sets available on request).

## File naming

`<what-it-shows>__<run-id>__<agent>-<post-id>.jsonl`

| File prefix | Agent | What to look for |
|---|---|---|
| `clean__…forecast-…` | the ModCast forecaster | a full investigation: dossier → hypothesis queries → precedent reads → verified submission |
| `repair__…forecast-…` | the forecaster | a **citation-verification rejection**: the verifier bounces unsupported factors back ("Citation verification FAILED…") and the agent repairs its submission — the "retries" the deliverable asks for |
| `induction__…` | the rulebook-induction agent | hypotheses proposed and tested with `compare_removal_rate`; after `finalize_rulebook`, code re-verifies every entry (dropped entries visible in `rulebooks/` provenance) |
| `oneshot__…` | the baseline chatbot | one prompt, one answer — what today's practice looks like |
| `oneshot_informed__…` | the informed baseline | same, with rules + base rate included in the prompt (changelog #10) |
| `predict_demo__…` | the forecaster in product mode | a real draft forecast via the CLI wizard |

## Event format (one JSON object per line)

- `{"event": "user", "payload": …}` — content sent to the model (the first one is the
  evidence dossier; later ones are tool results)
- `{"event": "assistant", "payload": [blocks]}` — the model's turn; blocks of type
  `tool_use` show each tool call with its name and full input (e.g. the exact `where_sql`
  hypothesis being tested); `text` blocks are commentary
- `{"event": "tool_results", "payload": […]}` — what each tool returned, matched by
  `tool_use_id`; `is_error: true` marks rejected queries (the model reads the error and retries)
- `{"event": "usage", …}` — per-turn model id, token counts, stop reason

Follow a file top to bottom and you see the whole loop the deliverable asks for: the agent's
instructions → what it did → how its tools responded → the feedback that shaped its next
step → the verification round → the final submission. Human checkpoints: none are needed at
forecast time by design — the human is the poster, who decides what to do with the forecast.
