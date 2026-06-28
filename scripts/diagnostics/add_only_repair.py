"""Q7 (POMDP-QP-FAR): D_quorum-guided add-only repair on anchor-FAILURE scenes (eval-only).

For each frame where the deployable local_hysteresis anchor is INFEASIBLE (true C < tau), run the CENTRAL
greedy D_quorum-guided add repair and measure whether it raises feasibility above the anchor while
retaining every anchor edge. Final metric = true closed-form PBFT C; D_quorum is the auxiliary guide.
Grouped: anchor (deployable, 0-eval) vs greedy repair (central reference, uses the evaluator).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from marl_topology.policies.decentralized_mutual_acceptance import local_mutual_assemble  # noqa: E402
from marl_topology.training.dynamic_baselines import local_hysteresis_action  # noqa: E402
from marl_topology.training.dynamic_rl import _budgets_edges  # noqa: E402
from marl_topology.training.quorum_deficit_bridge import topology_reliability  # noqa: E402
from marl_topology.training.residual_repair import greedy_dquorum_add_repair  # noqa: E402
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
    ap.add_argument("--max-adds", type=int, default=6)
    ap.add_argument("--seed", type=int, default=4707)
    ap.add_argument("--out", default=str(ROOT / "result_save" / "add_only_repair.json"))
    args = ap.parse_args()

    report = {"scope": "Q7 D_quorum-guided add-only repair on anchor-failure frames (eval-only)",
              "tau": TAU, "config": {"scenes": args.scenes, "frames": args.frames,
                                     "dyn_nodes": args.dyn_nodes, "max_adds": args.max_adds,
                                     "seed": args.seed}, "by_data": {}}
    for data in args.data:
        scenes = _build(data, args.seed * 1000 + 1, args.scenes, args)
        n_frames = n_fail = repaired = repairable = 0
        c_imp = d_red = added = ret = calls = 0.0
        for sc in scenes:
            prev: list = []
            for t in range(sc.n_frames):
                obs = sc.observation(t, prev)
                budgets, edges = _budgets_edges(obs["context"])
                anchor = local_hysteresis_action(obs["ef"], obs["edge_ids"], edges, budgets, prev,
                                                 keep_threshold=0.4, add_threshold=0.6)
                n_frames += 1
                ev = obs["context"].evaluator
                c_anchor = topology_reliability(ev, anchor)["consensus"]
                if c_anchor < TAU:                                   # an anchor-FAILURE frame
                    n_fail += 1
                    # NAIVE max-add baseline: every node accepts its top-budget edges (mutual). NOT an
                    # upper bound -- targeted D_quorum repair can beat it (adding the RIGHT edges, not
                    # the MOST). Reported to show guidance matters.
                    naive_topo = local_mutual_assemble([1.0] * len(obs["edge_ids"]), obs["edge_ids"], obs["context"])
                    repairable += int(topology_reliability(ev, naive_topo)["consensus"] >= TAU)
                    r = greedy_dquorum_add_repair(ev, anchor, obs["edge_ids"], obs["context"],
                                                  tau=TAU, max_adds=args.max_adds)
                    repaired += int(r["repaired_feasible"])
                    c_imp += r["C_improvement"]; d_red += r["d_quorum_reduction"]
                    added += r["n_added"]; ret += r["retention"]; calls += r["evaluator_calls"]
                prev = anchor                                        # the anchor rolls (deployable)
        f = max(1, n_fail)
        row = {"n_frames": n_frames, "anchor_failures": n_fail,
               "anchor_failure_rate": round(n_fail / max(1, n_frames), 4),
               "naive_max_add_feasible": repairable,                 # all-top-budget topology (not a bound)
               "naive_max_add_feasible_rate": round(repairable / f, 4),
               "repaired_to_feasible": repaired,
               "repair_success_rate": round(repaired / f, 4),
               "mean_C_improvement": round(c_imp / f, 5),
               "mean_d_quorum_reduction": round(d_red / f, 5),
               "mean_added_edges": round(added / f, 3),
               "mean_retention": round(ret / f, 4),
               "central_evaluator_calls_per_repair": round(calls / f, 1)}
        report["by_data"][data] = row
        print(f"[{data}] frames={n_frames} fail={n_fail} naive_max_add_feasible={repairable} "
              f"repaired={repaired} (rate={row['repair_success_rate']}) "
              f"C_imp={row['mean_C_improvement']} D_red={row['mean_d_quorum_reduction']} "
              f"added={row['mean_added_edges']} retention={row['mean_retention']} "
              f"evals/repair={row['central_evaluator_calls_per_repair']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
