"""R5 generator: multi-seed supervised edit-head learning -- does the held top-k beat random? (>=5 seeds, CI)

The pivotal deployable-learning test: train the repair/safety/edit heads on LOCAL features (no evaluator) to
predict the R4 beneficial edits, and report the held top-k edit precision vs the same-budget random base rate
with a 95% CI on (precision - base). precision - base CI > 0 => local features CAN predict the central
beneficial edits (KEEP -> R6); CI spans/below 0 => deployable LEARNING gap (STOP; the campaign's honest close).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))

import residual_ppo_train as rp3  # noqa: E402
from marl_topology.training.edit_head_training import train_edit_heads  # noqa: E402


class _A:
    dyn_nodes = [8, 12, 16]
    frames = 4
    gamma = 0.95


def _ci95(xs):
    xs = [x for x in xs if x is not None and x == x]
    n = len(xs)
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    m = sum(xs) / n
    if n < 2:
        return {"mean": round(m, 5), "lo": round(m, 5), "hi": round(m, 5), "n": n}
    sd = math.sqrt(sum((v - m) ** 2 for v in xs) / (n - 1))
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 2.776)
    h = t * sd / math.sqrt(n)
    return {"mean": round(m, 5), "lo": round(m - h, 5), "hi": round(m + h, 5), "n": n}


def main() -> None:
    seeds = [0, 1, 2, 3, 4]
    T = rp3._load_trunk()
    report = {"scope": "R5 supervised edit/repair/safety heads (LOCAL features only) on R4 targets; 5 seeds x "
                       "{random,urban}; held top-k edit precision vs random base rate", "by_data": {}}
    for data in ["random", "urban"]:
        per = {"topk_precision": [], "random_base_rate": [], "precision_minus_base": [], "lift": [],
               "repair_corr": [], "safety_corr": [], "untrained_precision_minus_base": []}
        for s in seeds:
            tr = rp3._build(data, s * 1000 + 1, 5, _A())
            hd = rp3._build(data, s * 1000 + 777, 10, _A())
            r = train_edit_heads(tr, hd, T, epochs=40, hidden=64, seed=s)
            t, u = r["trained_held"], r["untrained_held"]
            per["topk_precision"].append(t["topk_precision"])
            per["random_base_rate"].append(t["random_base_rate"])
            per["precision_minus_base"].append(t["precision_minus_base"])
            per["lift"].append(t["lift"])
            per["repair_corr"].append(t["repair_corr"])
            per["safety_corr"].append(t["safety_corr"])
            per["untrained_precision_minus_base"].append(u["precision_minus_base"])
        report["by_data"][data] = {k: _ci95(v) for k, v in per.items()}
        d = report["by_data"][data]
        beats = d["precision_minus_base"]["lo"] is not None and d["precision_minus_base"]["lo"] > 0
        report["by_data"][data]["heads_beat_random"] = beats
        print(f"[{data}] topk_prec {d['topk_precision']['mean']} base {d['random_base_rate']['mean']} | "
              f"prec-base {d['precision_minus_base']['mean']} CI[{d['precision_minus_base']['lo']},"
              f"{d['precision_minus_base']['hi']}] beats_random={beats} | repair_corr {d['repair_corr']['mean']} "
              f"safety_corr {d['safety_corr']['mean']}")
    out = ROOT / "result_save" / "belief_residual" / "R5" / "edit_heads_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
