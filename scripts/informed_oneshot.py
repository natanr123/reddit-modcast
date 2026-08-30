"""The missing ablation rung: one-shot + user-accessible context, no tools."""
import json

import numpy as np

from modcast import config, evaluate as E
from modcast.evaluate import paired_bootstrap, metrics
from modcast.llm import new_run_id
from modcast.llm_predictors import InformedOneShotPredictor
from modcast.store import Store

store = Store(read_only=True)
final = json.load(open("tmp/generated/results/llm/eval_latest.json"))
ids = final["test"]["post_ids"]
# select the exact evaluated posts by id: immune to corpus drift between fetches
from modcast.evaluate import _fetch  # reuse the record constructor fields
from modcast.schema import PostRecord
FIELDS = list(PostRecord.__dataclass_fields__)
rows = store.query(
    f"SELECT {', '.join(FIELDS)} FROM posts WHERE id IN ({','.join('?' * len(ids))})", ids
).fetchall()
by_id = {r[0]: PostRecord(**dict(zip(FIELDS, r))) for r in rows}
missing = [i for i in ids if i not in by_id]
assert not missing, f"posts missing from restored corpus: {missing}"
sub = [by_id[i] for i in ids]
y = np.array([1.0 if r.label == "removed_mod" else 0.0 for r in sub])

pred = InformedOneShotPredictor(run_id=new_run_id("informed"), con=store.con)
p = pred.predict_proba(sub)
m = metrics(y, p)
print("INFORMED_ONESHOT:", json.dumps(m))

others = {name: np.array(v["predictions"]) for name, v in final["predictors"].items()}
comp = paired_bootstrap(y, {**others, "llm_oneshot_informed": p})
keep = [c for c in comp if "llm_oneshot_informed" in (c["a"], c["b"])]
out = {"metrics": m, "comparisons": keep}
open("tmp/generated/results/informed_oneshot.json", "w").write(json.dumps(out, indent=1))
for c in keep:
    print(f"{c['a']} vs {c['b']}: {c['brier_a']} vs {c['brier_b']} adv={c['advantage_a']:+.4f} "
          f"CI={c['ci95']} p={c['p_a_not_better']}")
print("INFORMED_DONE")
