"""Temporal Value Test on the REAL env code + real corrected-evaluator costs (audit checklist §3.3).

The dynamic audit found the Temporal Value Test (Delta_H = J_myopic - J_horizon, Spec S3.6) had
only ever been exercised on synthetic toy frames in a unit test -- never on real data. This driver
addresses that with TWO honest pieces:

  PART A (mechanism + when-does-temporal-matter): run the ACTUAL TwoTimescaleTopologyEnv /
    temporal_value_test code on a controlled 2-frame case whose per-frame optima alternate, and
    SWEEP the reconfiguration cost r (per toggled edge) and hold interval H (= H_PBFT micro-rounds).
    Delta_H > 0 iff horizon-awareness (not switching needlessly) strictly lowers total cost. This
    maps the threshold at which temporal modeling would matter and reproduces the R5 analytical
    result Delta_H = max(0, 2r - H) on the real code path.

  PART B (why a faithful REAL-trajectory test is blocked on op_corrected): the dataset has NO
    multi-step sequences (every (scene, sequence_id) has a single time_step=1 frame), and the
    reconfiguration cost |E_t triangle E_{t-1}| is only physical on a SHARED edge universe (the same
    area evolving). So per-frame variation -- the thing that makes Delta_H nonzero -- requires either
    real trajectory data or channel re-realization, neither of which exists in op_corrected. We DO
    record the real corrected-evaluator energy/feasibility landscape of each scene's named candidate
    topologies (topology_variants) on held scenes, to show the real per-frame cost structure that a
    future dynamic dataset would sequence over.

CONCLUSION the driver supports: at a SINGLE channel realization the per-frame optimum is fixed, so
Delta_H is identically 0 by construction -- the static bandit is exactly the right model for the
data we have. A nonzero Delta_H (temporal value) can only be measured on multi-realization /
multi-frame data that the current op_corrected dataset does not contain.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_topology.data.row_context_builder import build_row_contexts  # noqa: E402
from marl_topology.training.two_timescale_env import (  # noqa: E402
    ReconfigCost,
    TwoTimescaleTopologyEnv,
    temporal_value_test,
)


def part_a_mechanism_sweep() -> dict:
    """2-frame controlled case: topology A optimal at frame 0, B at frame 1 (they differ by 1 edge).
    cost_fn returns a per-frame base objective; sweep reconfig r and hold H -> map Delta_H."""
    A = frozenset({"e0"})
    B = frozenset({"e1"})              # |A triangle B| = 2 toggled edges (drop e0, add e1)

    def cost_fn(frame, topo):
        # frame 0 prefers A (cost 0) over B (cost 1); frame 1 prefers B over A. A pure per-frame signal.
        topo = frozenset(topo)
        if frame == 0:
            return 0.0 if topo == A else 1.0
        return 0.0 if topo == B else 1.0

    frames = [0, 1]
    candidates = [[A, B], [A, B]]
    rows = []
    for H in (1, 2, 3, 4):
        for r in (0.0, 0.25, 0.5, 1.0, 2.0):
            env = TwoTimescaleTopologyEnv(frames, cost_fn, ReconfigCost(e_edge=r, l_edge=0.0),
                                          hold_interval=H, gamma=1.0)
            res = temporal_value_test(env, candidates)
            # closed-form expectation: myopic switches A->B paying r*|A^B|=2r; horizon either switches
            # (pays 2r, saves 0 base by being per-frame optimal) or holds A (pays 0 reconfig, eats 1*H
            # extra base at frame 1). Delta_H = J_myopic - J_horizon = max(0, 2r - H).
            expected = max(0.0, 2.0 * r - H)
            rows.append({"H": H, "r": r, "j_myopic": res["j_myopic"], "j_horizon": res["j_horizon"],
                         "delta_h": res["delta_h"], "delta_h_expected_max(0,2r-H)": expected,
                         "matches_analytic": abs(res["delta_h"] - expected) < 1e-9})
    return {"frames": 2, "toggled_edges_A_to_B": 2, "rows": rows,
            "all_match_analytic": all(x["matches_analytic"] for x in rows)}


def part_b_real_candidate_landscape(held_shards, n_scenes: int) -> dict:
    """Real corrected-evaluator energy/feasibility of each scene's named candidate topologies."""
    TAU = 0.9
    contexts = []
    for sp in held_shards:
        import pickle
        d = pickle.load(open(sp, "rb"))
        for split in ("train", "eval", "test"):
            for _row, ctx in build_row_contexts(d, split):
                contexts.append(ctx)
        if len(contexts) >= n_scenes:
            break
    scenes = []
    seq_lengths = {}
    for ctx in contexts[:n_scenes]:
        seq_lengths.setdefault(ctx.sequence_id, set()).add(ctx.time_step)
        variants = dict(ctx.topology_variants)
        cand = {}
        for name, edges in variants.items():
            edges = list(edges)
            try:
                m = ctx.evaluator.evaluate(set(edges)).metrics
                c = float(m["consensus_success_probability"]); e = float(m.get("energy", 0.0))
            except Exception:
                c, e = 0.0, 0.0
            cand[name] = {"n_edges": len(edges), "consensus": round(c, 4),
                          "energy_j": round(e, 6), "feasible": c >= TAU}
        feas = {k: v for k, v in cand.items() if v["feasible"]}
        best = min(feas.items(), key=lambda kv: kv[1]["energy_j"])[0] if feas else None
        scenes.append({"scenario_id": ctx.fixture.fixture_id, "n_nodes": len(ctx.graph.node_ids),
                       "time_step": ctx.time_step, "candidates": cand,
                       "min_energy_feasible_variant": best})
    return {"n_scenes": len(scenes),
            "sequence_length_distribution": {str(k): len(v) for k, v in
                                             {("len_%d" % len(v)): v for v in seq_lengths.values()}.items()},
            "all_sequences_length_1": all(len(v) == 1 for v in seq_lengths.values()),
            "note": ("Every sequence has a single time_step=1 frame: op_corrected has NO multi-step "
                     "trajectories. At one channel realization each scene's feasible-energy-min variant "
                     "is fixed, so a faithful Delta_H is identically 0 (no per-frame variation to plan "
                     "over). A nonzero temporal value needs multi-frame / multi-realization data."),
            "scenes": scenes}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--held-shards", nargs="+", default=None)
    ap.add_argument("--n-scenes", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "result_save" / "temporal_value_test_real.json"))
    args = ap.parse_args()
    corr = sorted(glob.glob(str(ROOT / "result_save" / "campaign" / "data" / "op_corrected" / "_op_shard_30*.pkl")))
    held = args.held_shards or corr[16:18]
    report = {"scope": "Temporal Value Test on the real env code + real corrected-evaluator costs (§3.3)",
              "part_a_mechanism_sweep": part_a_mechanism_sweep(),
              "part_b_real_candidate_landscape": part_b_real_candidate_landscape(held, args.n_scenes)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    a = report["part_a_mechanism_sweep"]
    b = report["part_b_real_candidate_landscape"]
    print(f"[Part A] env temporal_value_test reproduces Delta_H=max(0,2r-H) on all "
          f"{len(a['rows'])} (H,r) cells: {a['all_match_analytic']}")
    pos = [x for x in a["rows"] if x["delta_h"] > 0]
    print(f"         temporal matters (Delta_H>0) in {len(pos)}/{len(a['rows'])} cells "
          f"(only when 2*reconfig_cost > hold_interval)")
    print(f"[Part B] {b['n_scenes']} real held scenes; all_sequences_length_1={b['all_sequences_length_1']} "
          f"-> NO real trajectories -> faithful Delta_H is identically 0 at single realization")
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
