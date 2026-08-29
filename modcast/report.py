"""Preflight report: the user-facing artifact for a single candidate post."""
from __future__ import annotations

from modcast.agent import Forecast
from modcast.schema import PostRecord


def _band(p: float) -> str:
    if p < 0.15:
        return "CLEAR SKIES"
    if p < 0.35:
        return "LIGHT CLOUDS"
    if p < 0.60:
        return "STORM WATCH"
    return "SEVERE WEATHER"


def render(record: PostRecord, fc: Forecast, base_rate: float | None = None) -> str:
    lines = [
        f"# ModCast preflight report — r/{record.subreddit}",
        "",
        f"**Removal risk: {fc.p_removed:.0%} — {_band(fc.p_removed)}**"
        + (f" (subreddit base rate: {base_rate:.0%})" if base_rate is not None else ""),
        "",
        f"> {record.title}",
        "",
    ]
    if fc.risk_factors:
        lines.append("## Risk factors (every citation machine-verified)")
        for f in fc.risk_factors:
            arrow = "▲" if f.get("direction") == "increases" else "▼"
            lines.append(f"- {arrow} {f['factor']}")
            if f.get("rule_ref"):
                lines.append(f"  - rule: {f['rule_ref']}")
            ids = ", ".join(f.get("evidence_post_ids", []))
            lines.append(f"  - precedent: {ids}")
    else:
        lines.append("## Risk factors\n- none verified")
    if fc.suggested_fixes:
        lines.append("\n## Suggested fixes")
        lines += [f"{i}. {s}" for i, s in enumerate(fc.suggested_fixes, 1)]
    lines += ["", "## Forecast reasoning", fc.reasoning_summary]
    if fc.rejected_factors:
        lines.append("\n## Discarded claims (failed citation verification)")
        lines += [f"- {f['factor']} — {f['rejection']}" for f in fc.rejected_factors]
    return "\n".join(lines) + "\n"
