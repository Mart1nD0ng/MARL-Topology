"""R4 generator: multi-seed beneficial oracle-edit dataset metrics + CI (Contract v4: >=5 seeds).

Builds the beneficial-edit teacher dataset over the anchor for 5 seeds x {urban, random} and reports the
positive_edit_rate / anchor_failure_repairable_rate / safe_prune_rate / best_edit_gain with 95% CI -- the
pivotal measurement of whether a learnable beneficial-edit DIRECTION signal exists over the local_hysteresis
anchor at N<=16 (the lever R3 showed the trainer lacks). Teacher-only (train scenes; never held).
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
from marl_topology.training.oracle_edit_dataset import build_edit_dataset, supervised_records  # noqa: E402


class _A:
    dyn_nodes = [8, 12, 16]
    frames = 5
    gamma = 0.95


def _ci95(xs):
    xs = [x for x in xs if x is not None]
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
    report = {"scope": "R4 beneficial oracle-edit dataset; 5 seeds x {random,urban}; teacher-only; ΔJ in the "
                       "dense reward; positive = ΔJ>margin AND safe", "by_data": {}}
    for data in ["random", "urban"]:
        per = {"positive_edit_rate": [], "positive_edit_rate_m01": [], "anchor_failure_repairable_rate": [],
               "safe_prune_rate": [], "best_edit_gain_max": [], "n_records": [], "evaluator_calls": []}
        hashes = []
        for s in seeds:
            sc = rp3._build(data, s * 1000 + 1, 5, _A())
            rec, m = build_edit_dataset(sc, T, margin=0.0)
            per["positive_edit_rate"].append(m["positive_edit_rate"])
            per["positive_edit_rate_m01"].append(round(len(supervised_records(rec, margin=0.01)) / max(1, m["n_records"]), 5))
            per["anchor_failure_repairable_rate"].append(
                m["anchor_failure_repairable_rate"] if m["n_anchor_infeasible_frames"] > 0 else None)
            per["safe_prune_rate"].append(
                m["safe_prune_rate"] if m["n_anchor_feasible_frames"] > 0 else None)
            per["best_edit_gain_max"].append(m["best_edit_gain_max"])
            per["n_records"].append(m["n_records"]); per["evaluator_calls"].append(m["evaluator_calls"])
            hashes.append(m["dataset_hash"])
        report["by_data"][data] = {k: _ci95(v) for k, v in per.items()}
        report["by_data"][data]["dataset_hashes"] = hashes
        d = report["by_data"][data]
        print(f"[{data}] positive_edit_rate {d['positive_edit_rate']['mean']} "
              f"CI[{d['positive_edit_rate']['lo']},{d['positive_edit_rate']['hi']}] | "
              f"repairable {d['anchor_failure_repairable_rate']['mean']} | "
              f"safe_prune {d['safe_prune_rate']['mean']} | best_gain {d['best_edit_gain_max']['mean']}")
    out = ROOT / "result_save" / "belief_residual" / "R4" / "oracle_edit_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
