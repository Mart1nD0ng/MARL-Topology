"""Q8 (POMDP-QP-FAR): D_quorum-safety-guided conservative prune on anchor-FEASIBLE scenes (eval-only).

For each frame where the deployable local_hysteresis anchor is FEASIBLE (true C >= tau), greedily REMOVE
the lowest-risk edge (risk_e = D(x∖e)-D(x)) that keeps feasibility, and measure the true energy/latency
change. Final metrics = true closed-form PBFT C / energy / latency; D_quorum is the auxiliary safety guide.
Central reference (uses the evaluator); reports feasibility retention + cost direction + critical-edge
deletions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from marl_topology.training.dynamic_baselines import local_hysteresis_action  # noqa: E402
from marl_topology.training.dynamic_rl import _budgets_edges  # noqa: E402
from marl_topology.training.quorum_deficit_bridge import topology_reliability  # noqa: E402
from marl_topology.training.residual_repair import greedy_conservative_prune  # noqa: E402
from marl_topology.training.two_timescale_env import ReconfigCost  # noqa: E402

TAU = 0.9


def _build(data, seed, count, args):
    from build_operating_point_dataset import operating_point_regime
    from marl_topology.training.dynamic_frames import (
        sample_dynamic_scenes, sample_dynamic_urban_scenes)
    common = dict(seed=seed, count=count, node_count_choices=tuple(args.dyn_nodes),
                  regime=operating_point_regime(args.tx_power), num_frames=args.frames, dt_s=2.0,
                  speed_min_mps=15.0, speed_max_mps=30.0, reconfig=ReconfigCost(),
                  hold_interval=4, gamma=0.95)
    if data == "urban":
        return sample_dynamic_urban_scenes(rsu_count=4, blocks_per_side=3, **common)
    return sample_dynamic_scenes(**common)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["random", "urban"])
    ap.add_argument("--scenes", type=int, default=8)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--dyn-nodes", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--tx-power", type=float, default=20.0)
    ap.add_argument("--max-removes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=4808)
    ap.add_argument("--out", default=str(ROOT / "result_save" / "conservative_prune.json"))
    args = ap.parse_args()

    report = {"scope": "Q8 D_quorum-safety conservative prune on anchor-feasible frames (eval-only)",
              "tau": TAU, "config": {"scenes": args.scenes, "frames": args.frames,
                                     "dyn_nodes": args.dyn_nodes, "max_removes": args.max_removes,
                                     "seed": args.seed}, "by_data": {}}
    for data in args.data:
        scenes = _build(data, args.seed * 1000 + 1, args.scenes, args)
        n_frames = n_feas = retained = crit = 0
        e_red = l_red = removed = c_chg = calls = 0.0
        e_down = 0
        for sc in scenes:
            prev: list = []
            for t in range(sc.n_frames):
                obs = sc.observation(t, prev)
                budgets, edges = _budgets_edges(obs["context"])
                anchor = local_hysteresis_action(obs["ef"], obs["edge_ids"], edges, budgets, prev,
                                                 keep_threshold=0.4, add_threshold=0.6)
                n_frames += 1
                ev = obs["context"].evaluator
                if topology_reliability(ev, anchor)["consensus"] >= TAU and len(anchor) >= 2:
                    n_feas += 1
                    r = greedy_conservative_prune(ev, anchor, obs["edge_ids"], obs["context"],
                                                  tau=TAU, max_removes=args.max_removes)
                    retained += int(r["feasibility_retained"]); crit += r["critical_edge_deletions"]
                    e_red += r["energy_reduction"]; l_red += r["latency_reduction"]
                    removed += r["n_removed"]; calls += r["evaluator_calls"]
                    c_chg += (r["pruned_C"] - r["anchor_C"])
                    e_down += int(r["energy_reduction"] > 1e-9)
                prev = anchor
        f = max(1, n_feas)
        row = {"n_frames": n_frames, "anchor_feasible": n_feas,
               "feasibility_retention": round(retained / f, 4),
               "critical_edge_deletions": int(crit),
               "mean_energy_reduction": round(e_red / f, 6),       # POSITIVE = energy went DOWN
               "frac_energy_reduced": round(e_down / f, 4),
               "mean_latency_reduction": round(l_red / f, 6),
               "mean_removed_edges": round(removed / f, 3),
               "mean_C_change": round(c_chg / f, 5),
               "central_evaluator_calls_per_prune": round(calls / f, 1)}
        report["by_data"][data] = row
        print(f"[{data}] frames={n_frames} feasible={n_feas} retention={row['feasibility_retention']} "
              f"crit_del={row['critical_edge_deletions']} mean_E_reduction={row['mean_energy_reduction']} "
              f"(frac_down={row['frac_energy_reduced']}) mean_L_reduction={row['mean_latency_reduction']} "
              f"removed={row['mean_removed_edges']} dC={row['mean_C_change']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
