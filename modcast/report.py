"""Preflight report: the user-facing artifact for a single candidate post."""
from __future__ import annotations

import re
import time

import typer

from modcast.agent import Forecast
from modcast.schema import PostRecord

_BAND_COLORS = {
    "CLEAR SKIES": "green",
    "LIGHT CLOUDS": "yellow",
    "STORM WATCH": "yellow",
    "SEVERE WEATHER": "red",
}

# similar-posts table: truncation budgets so a row fits a normal terminal
NEIGHBORS_SHOWN = 8
TITLE_CHARS = 38
BODY_CHARS = 40


def _cell(text: str, limit: int) -> str:
    """One markdown table cell: newlines and pipes collapsed, truncated."""
    t = " ".join((text or "").split()).replace("|", "¦")
    return t[: limit - 1] + "…" if len(t) > limit else t


def colorize(text: str) -> str:
    """ANSI-style the rendered report for terminal display.

    The saved .md stays plain; typer.echo strips these codes automatically
    when output is piped or redirected.
    """
    out = []
    for line in text.splitlines():
        if line.startswith("**Removal risk:"):
            color = next((c for band, c in _BAND_COLORS.items() if band in line), None)
            line = typer.style(line, fg=color, bold=True)
        elif line.startswith("## ⚠"):
            line = typer.style(line, fg="red", bold=True)
        elif line.startswith("## ✓"):
            line = typer.style(line, fg="green", bold=True)
        elif line.startswith("## Suggested fixes"):
            line = typer.style(line, fg="yellow", bold=True)
        elif line.startswith("- ▲"):
            line = typer.style(line, fg="red")
        elif line.startswith("- ▼"):
            line = typer.style(line, fg="green")
        elif line.startswith("## Discarded claims"):
            line = typer.style(line, dim=True)
        elif re.match(r"^\| \d+% +\|", line):
            line = typer.style(line, fg="red" if "| removed |" in line else "green")
        out.append(line)
    return "\n".join(out)


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
    def _render(factors: list[dict], arrow: str) -> list[str]:
        out = []
        for f in factors:
            out.append(f"- {arrow} {f['factor']}")
            if f.get("rule_ref"):
                out.append(f"  - rule: {f['rule_ref']}")
            out.append(f"  - precedent: {', '.join(f.get('evidence_post_ids', []))}")
        return out

    risks = [f for f in fc.risk_factors if f.get("direction") == "increases"]
    protective = [f for f in fc.risk_factors if f.get("direction") != "increases"]
    if risks:
        lines.append("## ⚠ Risk factors (every citation machine-verified)")
        lines += _render(risks, "▲")
    if protective:
        if risks:
            lines.append("")
        lines.append("## ✓ Working in your favor (every citation machine-verified)")
        lines += _render(protective, "▼")
    if not fc.risk_factors:
        lines.append("## Risk factors\n- none verified")
    if fc.suggested_fixes:
        lines.append("\n## Suggested fixes")
        lines += [f"{i}. {s}" for i, s in enumerate(fc.suggested_fixes, 1)]
    lines += ["", "## Forecast reasoning", fc.reasoning_summary]
    if fc.rejected_factors:
        lines.append("\n## Discarded claims (failed citation verification)")
        lines += [f"- {f['factor']} — {f['rejection']}" for f in fc.rejected_factors]
    if fc.neighbors:
        widths = (4, 7, 7, 8, TITLE_CHARS, BODY_CHARS)
        headers = ("sim", "fate", "when", "id", "title", "body")

        def _row(cells):
            return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

        lines += ["", "## Similar past posts (ids open at redd.it/<id>)", "",
                  _row(headers), _row(tuple("-" * w for w in widths))]
        for nb in fc.neighbors[:NEIGHBORS_SHOWN]:
            fate = "removed" if nb["label"] == "removed_mod" else "kept"
            when = time.strftime("%Y-%m", time.gmtime(nb["created_utc"]))
            lines.append(_row((f"{nb['score']:.0%}", fate, when, nb["id"],
                               _cell(nb.get("title", ""), TITLE_CHARS),
                               _cell(nb.get("body", ""), BODY_CHARS))))
        ns = fc.neighbor_summary
        if ns.get("k", 0) > len(fc.neighbors[:NEIGHBORS_SHOWN]) and ns.get("rate") is not None:
            lines.append(f"\n_showing {min(len(fc.neighbors), NEIGHBORS_SHOWN)} of {ns['k']} retrieved; "
                         f"removal rate among all {ns['k']}: {ns['rate']:.0%}_")
    return "\n".join(lines) + "\n"
