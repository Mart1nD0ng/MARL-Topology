"""R7 generator: adaptive anchor-KL vs R3-fixed anchor pull -- does an ADAPTIVE trust region beat the anchor?

Arms (same BeliefResidualActor / rollout / CTDE critic / PPO; ONE variable = the anchor pull):
  fixed     R3 baseline: residual_prior=-1.0 (fixed pull), no adaptive anchor-KL
  adaptive  R7: residual_prior=0.0 + adaptive anchor-KL (beta tightens on retention-drop / loosens on val-up)
Reports per seed + 95% CI on (adaptive residual_feas - anchor_feas) and (adaptive - fixed residual_feas), plus
edit_rate / retention / beta_anchor trajectory / approx_kl / EV / diverged. Expected per the R6 sweep (optimal
anchor-deviation=0): adaptive tightens to == anchor -> CONFIRMATORY negative. A CI > 0 would be a positive.
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
    report = {"scope": "R7 adaptive anchor-KL vs R3-fixed anchor pull; deployed MAP decode true PBFT feas; "
                       "5 seeds x {random,urban}; (adaptive residual - anchor) and (adaptive - fixed) feas CI",
              "by_data": {}}
    for data in ["random", "urban"]:
        per = {"fixed_resid_feas": [], "adaptive_resid_feas": [], "anchor_feas": [],
               "adaptive_minus_anchor": [], "adaptive_minus_fixed": [], "adaptive_edit_rate": [],
               "adaptive_retention": [], "fixed_edit_rate": [], "beta_final": [], "adaptive_diverged": []}
        beta_trajs = []
        for s in seeds:
            train = rp3._build(data, s * 1000 + 1, 6, _A())
            fx = rp3.train_residual_ppo(train, epochs=20, ppo_epochs=4, seed=s, T=T,
                                        adaptive_anchor_kl=False, residual_prior=-1.0)
            ad = rp3.train_residual_ppo(train, epochs=20, ppo_epochs=4, seed=s, T=T,
                                        adaptive_anchor_kl=True, residual_prior=0.0)
            per["fixed_resid_feas"].append(fx["residual_feasibility"])
            per["adaptive_resid_feas"].append(ad["residual_feasibility"])
            per["anchor_feas"].append(ad["anchor_feasibility"])
            per["adaptive_minus_anchor"].append(ad["residual_feasibility"] - ad["anchor_feasibility"])
            per["adaptive_minus_fixed"].append(ad["residual_feasibility"] - fx["residual_feasibility"])
            per["adaptive_edit_rate"].append(ad["edit_rate"])
            per["adaptive_retention"].append(ad["retention"])
            per["fixed_edit_rate"].append(fx["edit_rate"])
            per["beta_final"].append(ad["beta_anchor_final"])
            per["adaptive_diverged"].append(1.0 if ad["diverged"] else 0.0)
            beta_trajs.append(ad["beta_anchor_history"])
        d = {k: _ci95(v) for k, v in per.items()}
        d["beats_anchor"] = d["adaptive_minus_anchor"]["lo"] is not None and d["adaptive_minus_anchor"]["lo"] > 0
        d["beta_trajectories"] = beta_trajs
        report["by_data"][data] = d
        print(f"[{data}] anchor_feas {d['anchor_feas']['mean']} adaptive_resid {d['adaptive_resid_feas']['mean']} "
              f"fixed_resid {d['fixed_resid_feas']['mean']} | adapt-anchor {d['adaptive_minus_anchor']['mean']} "
              f"CI[{d['adaptive_minus_anchor']['lo']},{d['adaptive_minus_anchor']['hi']}] beats={d['beats_anchor']} "
              f"| adapt_edit {d['adaptive_edit_rate']['mean']} adapt_ret {d['adaptive_retention']['mean']} "
              f"beta_final {d['beta_final']['mean']}")
    out = ROOT / "result_save" / "belief_residual" / "R7" / "adaptive_kl_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
