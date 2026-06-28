"""Q12 (POMDP-QP-FAR): the consolidated multi-seed campaign -- DEPLOYABLE vs CENTRAL, grouped.

Runs the rollout arms across >=5 seeds x {urban, random} x held-N{8,12,16}: DEPLOYABLE (local_hysteresis
= the anchor, local_threshold; 0 action-evaluator calls) and CENTRAL REFERENCE (myopic-greedy = the
oracle; uses the evaluator at action time). Reports per-seed + 95% CI + the evaluator-call budget,
grouped (Contract S10.1). The full-residual / +PBRS / +PNA arms EQUAL the anchor (Q9/Q11, paired diff
0.000) and are cited in the decision, not re-trained. Final metrics = true closed-form PBFT C/E/L.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from marl_topology.training.dynamic_baselines import (  # noqa: E402
    evaluate_central_reference, evaluate_deployable_baseline,
    local_hysteresis_action, local_threshold_action)
from marl_topology.training.two_timescale_env import ReconfigCost  # noqa: E402


def ci95(values):
    """Mean and 95% CI half-width (t.975 by df) for a small sample."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    m = sum(values) / n
    if n == 1:
        return {"mean": round(m, 5), "lo": round(m, 5), "hi": round(m, 5), "n": 1}
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}.get(n, 2.776)
    h = t * sd / math.sqrt(n)
    return {"mean": round(m, 5), "lo": round(m - h, 5), "hi": round(m + h, 5), "n": n}


def group_arms(arm_results):
    """Group per-arm results into deployable_policy / central_reference, asserting the deployable arms
    make 0 action-evaluator calls (Contract S10.1)."""
    groups: dict = {}
    for r in arm_results:
        groups.setdefault(r["group"], []).append(r)
    deployable_ok = all(r["action_evaluator_calls"] == 0 for r in groups.get("deployable_policy", []))
    return {"groups": groups, "deployable_use_no_evaluator_for_action": deployable_ok}


def _load_trunk():
    spec = importlib.util.spec_from_file_location(
        "trunk", ROOT / "scripts" / "train" / "train_decentralized_rl.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _arms_for_seed(seed, data, args, T):
    from build_operating_point_dataset import operating_point_regime
    from marl_topology.training.dynamic_frames import sample_dynamic_scenes, sample_dynamic_urban_scenes
    common = dict(seed=seed * 1000 + 777, count=args.held, node_count_choices=tuple(args.dyn_nodes),
                  regime=operating_point_regime(args.tx_power), num_frames=args.frames, dt_s=2.0,
                  speed_min_mps=15.0, speed_max_mps=30.0, reconfig=ReconfigCost(e_edge=0.05, l_edge=0.0),
                  hold_interval=4, gamma=0.95, tag="held_")
    scenes = (sample_dynamic_urban_scenes(rsu_count=4, blocks_per_side=3, **common) if data == "urban"
              else sample_dynamic_scenes(**common))

    def rof(obs, edges, e_ref, lc, lb, beta, rm):
        return T.reward_of(obs, edges, e_ref, lc, lb, beta, rm)

    def rfe(obs):
        try:
            _c, e, _l = T._evaluate(obs, obs["edge_ids"]); return max(e, 1e-9)
        except Exception:
            return 1e-9

    rw = dict(reward_of=rof, ref_energy=rfe, lam_c=1.0, lam_b=1.0, beta=0.1, reward_mode="dense")
    hys = evaluate_deployable_baseline(
        lambda ef, e, ed, b, p: local_hysteresis_action(ef, e, ed, b, p, keep_threshold=0.4, add_threshold=0.6),
        scenes, label="local_hysteresis(anchor)", **rw)
    thr = evaluate_deployable_baseline(
        lambda ef, e, ed, b, p: local_threshold_action(ef, e, ed, b, p, threshold=0.5),
        scenes, label="local_threshold", **rw)
    cen = evaluate_central_reference(scenes, label="myopic_greedy(oracle)", **rw)
    return [hys, thr, cen]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--data", nargs="+", default=["random", "urban"])
    ap.add_argument("--held", type=int, default=12)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--dyn-nodes", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--tx-power", type=float, default=20.0)
    ap.add_argument("--out", default=str(ROOT / "result_save" / "pomdp_qpfar_campaign.json"))
    args = ap.parse_args()

    T = _load_trunk()
    report = {"scope": "Q12 POMDP-QP-FAR consolidated campaign (deployable vs central, per-seed + CI)",
              "config": {"seeds": args.seeds, "held": args.held, "frames": args.frames,
                         "dyn_nodes": args.dyn_nodes},
              "cited_residual_arms": ("full-residual MLP / +PBRS / +PNA == anchor (Q9/Q11, 5-seed paired "
                                      "residual-anchor diff 0.000, CI [0,0]); add-repair Q7 (random 22% of "
                                      "anchor-failures) + prune Q8 (urban 47%) are CENTRAL references"),
              "by_data": {}}
    for data in args.data:
        per_seed = {}                                              # arm label -> list over seeds
        groups_seen = {}
        calls = {}
        for seed in args.seeds:
            arms = _arms_for_seed(seed, data, args, T)
            for r in arms:
                per_seed.setdefault(r["label"], {"feas": [], "ret": [], "sw": []})
                per_seed[r["label"]]["feas"].append(r["per_frame_feasibility"])
                per_seed[r["label"]]["ret"].append(r["mean_episode_return"])
                per_seed[r["label"]]["sw"].append(r["mean_switches_per_frame"])
                groups_seen[r["label"]] = r["group"]
                calls[r["label"]] = r["action_evaluator_calls"]
        rows = {}
        for label, d in per_seed.items():
            rows[label] = {"group": groups_seen[label], "action_evaluator_calls": calls[label],
                           "feasibility": ci95(d["feas"]), "episode_return": ci95(d["ret"]),
                           "switches_per_frame": ci95(d["sw"]), "feasibility_per_seed": [round(x, 4) for x in d["feas"]]}
        grouped = group_arms([{"label": k, "group": v["group"], "action_evaluator_calls": v["action_evaluator_calls"]}
                              for k, v in rows.items()])
        report["by_data"][data] = {"arms": rows,
                                   "deployable_use_no_evaluator_for_action": grouped["deployable_use_no_evaluator_for_action"]}
        print(f"=== {data} ({len(args.seeds)} seeds) ===")
        for label, row in rows.items():
            f = row["feasibility"]
            print(f"  [{row['group'][:6]}] {label:<26} feas={f['mean']:.3f} CI[{f['lo']:.3f},{f['hi']:.3f}] "
                  f"return={row['episode_return']['mean']:.2f} sw={row['switches_per_frame']['mean']:.2f} "
                  f"eval_calls={row['action_evaluator_calls']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
