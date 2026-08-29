"""Dollar-spend accounting from trajectory usage logs."""
from __future__ import annotations

import json
from pathlib import Path

from modcast.llm import TRAJ_DIR

# $ per 1M tokens: (input, output, cache_write, cache_read)
PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5": (5.0, 25.0, 6.25, 0.5),
    "claude-sonnet-5": (2.0, 10.0, 2.5, 0.2),
    "claude-haiku-4-5": (1.0, 5.0, 1.25, 0.1),
}


def _price(model: str) -> tuple[float, float, float, float]:
    for key, p in PRICES.items():
        if model.startswith(key):
            return p
    return PRICES["claude-opus-5"]  # unknown models priced pessimistically


def run_cost(run_id: str | None = None) -> dict:
    """Total cost (USD) for one run_id, or all runs when None."""
    total = {"usd": 0.0, "input": 0, "output": 0, "cache_write": 0, "cache_read": 0, "calls": 0}
    roots = [TRAJ_DIR / run_id] if run_id else [p for p in TRAJ_DIR.iterdir() if p.is_dir()] if TRAJ_DIR.exists() else []
    for root in roots:
        for f in root.glob("*.jsonl"):
            for line in f.read_text().splitlines():
                rec = json.loads(line)
                if rec.get("event") != "usage":
                    continue
                p = rec["payload"]
                pi, po, pw, pr = _price(p.get("model", ""))
                total["usd"] += (
                    p.get("input_tokens", 0) * pi
                    + p.get("output_tokens", 0) * po
                    + p.get("cache_creation_input_tokens", 0) * pw
                    + p.get("cache_read_input_tokens", 0) * pr
                ) / 1e6
                total["input"] += p.get("input_tokens", 0)
                total["output"] += p.get("output_tokens", 0)
                total["cache_write"] += p.get("cache_creation_input_tokens", 0)
                total["cache_read"] += p.get("cache_read_input_tokens", 0)
                total["calls"] += 1
    total["usd"] = round(total["usd"], 4)
    return total


if __name__ == "__main__":
    import sys

    print(json.dumps(run_cost(sys.argv[1] if len(sys.argv) > 1 else None), indent=1))
