"""R3 A/B: does residual PPO+critic fix the free-REINFORCE clamp/collapse? (Contract v4: >=5 seeds)

Same BeliefResidualActor (R1 +-3 head) + same rollout + same scenes/seeds -- the ONLY difference is the
optimizer: free REINFORCE (scalar moving baseline, no PPO/critic/KL) vs Residual PPO (per-edge clip + KL
early-stop + CTDE critic + entropy). Reports per-seed retention / residual-vs-anchor feas / edit_rate and
the collapse rate (retention < 0.5). Isolates the TRAINER as the variable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))

import residual_ppo_train as rp3  # noqa: E402


class _A:
    dyn_nodes = [8, 12, 16]
    frames = 6
    gamma = 0.95


def _collapsed(m):
    return m["retention"] < 0.5 or m["residual_feasibility"] < m["anchor_feasibility"] - 0.2


def main() -> None:
    seeds = [0, 1, 2, 3, 4]
    T = rp3._load_trunk()
    report = {"scope": "R3 A/B: free REINFORCE vs residual PPO+critic (same actor/rollout/scenes); "
                       "collapse = retention<0.5 or feas drop>0.2", "by_data": {}}
    for data in ["random", "urban"]:
        rows = {"reinforce": [], "ppo": []}
        for s in seeds:
            tr = rp3._build(data, s * 1000 + 1, 8, _A())
            re = rp3.train_residual_reinforce(tr, epochs=20, seed=s, residual_prior=-1.0, T=T)
            pp = rp3.train_residual_ppo(tr, epochs=20, ppo_epochs=4, target_kl=0.05, seed=s,
                                        residual_prior=-1.0, T=T)
            rows["reinforce"].append(re); rows["ppo"].append(pp)
        def summarize(ms):
            return {"collapse_seeds": f"{sum(_collapsed(m) for m in ms)}/{len(ms)}",
                    "diverged_seeds": f"{sum(m['diverged'] for m in ms)}/{len(ms)}",
                    "retention_per_seed": [m["retention"] for m in ms],
                    "residual_feas_per_seed": [m["residual_feasibility"] for m in ms],
                    "anchor_feas_per_seed": [m["anchor_feasibility"] for m in ms],
                    "edit_rate_per_seed": [m["edit_rate"] for m in ms]}
        report["by_data"][data] = {"reinforce": summarize(rows["reinforce"]), "ppo": summarize(rows["ppo"])}
        for arm in ("reinforce", "ppo"):
            d = report["by_data"][data][arm]
            print(f"  [{data:7} {arm:9}] collapse={d['collapse_seeds']} diverged={d['diverged_seeds']} "
                  f"retention={d['retention_per_seed']} resid_feas={d['residual_feas_per_seed']} "
                  f"edit={d['edit_rate_per_seed']}")
    out = ROOT / "result_save" / "belief_residual" / "R3" / "ppo_vs_reinforce.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
