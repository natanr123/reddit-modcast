# Agent trajectories

Every LLM interaction in ModCast is logged **as it happens** — one JSONL file per agent
conversation, capturing the instructions, every tool call, every tool response, the feedback
that shaped the next step, retries, and the final result. Curated representative files for
**every agent used** live in [`results/trajectories/`](results/trajectories/); this document
is the map.

## The agents, and their representative trajectories

| Agent | Instructions live in | Representative file (in `results/trajectories/`) |
|---|---|---|
| **ModCast forecaster** (the product) | `modcast/agent.py` — `SYSTEM` + `TOOLS` | `clean__…forecast-1vyg1qo.jsonl` — a full investigation |
| forecaster, **verification-retry case** | same | `repair__…forecast-1vxyf9g.jsonl` — a rejected citation and its repair (annotated below) |
| forecaster, **product mode** | same | `predict_demo__…forecast-draft-….jsonl` — a real CLI draft forecast |
| **Rulebook-induction agent** (verified memory; later removed from the forecaster per changelog #6) | `modcast/induce.py` — `SYSTEM` + `TOOLS` | `induction__…induce-personalfinance.jsonl` |
| **One-shot baseline** (today's practice) | `modcast/llm_predictors.py` — `OneShotPredictor` | `oneshot__…jsonl` |
| **Informed one-shot baseline** (context, no tools) | `modcast/llm_predictors.py` — `InformedOneShotPredictor` | `oneshot_informed__…jsonl` |

The full corpus — 5,000+ trajectories covering all 600-post evaluation arms — regenerates
with any eval command and is available on request (82 MB, excluded from the repo for size).

## Event format (one JSON object per line)

| `event` | Meaning |
|---|---|
| `user` | content sent to the model — the first is the evidence dossier; later ones are tool results |
| `assistant` | the model's turn; `tool_use` blocks show each call with its full input (e.g. the exact `where_sql` hypothesis); `text` blocks are commentary |
| `tool_results` | what each tool returned, matched by `tool_use_id`; `is_error: true` marks rejected inputs — the model reads the error and adapts |
| `usage` | per-turn model id, token counts, stop reason |

## Annotated walkthrough: the verification-retry case

`repair__eval-…__forecast-1vxyf9g.jsonl` (r/unpopularopinion, a post about sexual-harassment
discourse), event by event:

1. **Turn 1 — the agent investigates in parallel**: four `read_post` calls on the nearest
   precedents *and* three statistical hypotheses in the same turn — is missing flair
   predictive here? (86.7% removal when true, n=38,324), are very long titles? (91.3%,
   n=1,429), are titles mentioning sexual harassment? (**24/24 removed, 100%**, CI [86%, 100%]).
2. **Tool results return** — real rates with confidence intervals, computed by code, never
   by the model.
3. **First `submit_forecast` (p=0.95)** — among its risk factors, one about missing flair.
4. **The verifier rejects it**: `Citation verification FAILED for some factors: [{"factor":
   "The post has no link flair…"` — the cited precedent posts didn't support that specific
   claim. This is the automated "retry" checkpoint: nothing unverifiable may reach a report.
5. **Second `submit_forecast`** — the factor repaired with supportable evidence; `forecast
   accepted`. The final report shows only machine-verified factors (and lists discarded
   claims openly — see any saved report's "Discarded claims" section).

That loop — instructions → parallel evidence gathering → tool feedback → submission →
independent verification → repair → acceptance — is the system's core, and it is visible in
every forecast trajectory, not just this one.

## Human checkpoints

By design there are none at forecast time: ModCast is read-only (it never posts or acts on
Reddit), so the human checkpoint is the poster themselves, deciding what to do with the
forecast. During *development*, the human checkpoints were the measured go/no-go and
ablation decisions recorded in the README changelog.

## Coding-agent traces (building the project)

The project was built with Claude Code (Fable 5) and the Codex CLI as required by the
challenge; the trajectories above are the *solution's* agents. Development-time agent usage
is disclosed in the README ("Coding-agent disclosure") and development traces are available
on request.
