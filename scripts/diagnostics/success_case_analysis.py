"""Success-case analysis: did the deployable MARL catch up to the heuristic, and at what C/E/L?

The Q9/Q11 result is that the residual MARL policy == the local_hysteresis anchor EXACTLY (paired diff
0.000, retention 1.0, element-wise-identical feasibility) -- so the deployable MARL produces the SAME
topology as the heuristic on every frame, hence IDENTICAL true PBFT reliability/energy/latency. This
script makes that quantitative and contrasts both against the central myopic oracle: per held frame it
rolls out the anchor (= deployable MARL) and the oracle, records (C, E, L, feasible) via the TRUE
closed-form PBFT evaluator on the current channel, and classifies every frame as both-feasible /
anchor-only / oracle-only / both-infeasible. On the shared-success frames it reports whether the oracle
buys lower energy/latency; on the oracle-only frames it reports what the oracle recovers that the anchor
misses. 5 seeds x {urban, random} x held-N{8,12,16}. Eval-only; no production-path change.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from marl_topology.training.dynamic_baselines import local_hysteresis_action  # noqa: E402
from marl_topology.training.dynamic_rl import _budgets_edges  # noqa: E402
from marl_topology.training.two_timescale_env import ReconfigCost  # noqa: E402

TAU = 0.9


def _load_trunk():
    spec = importlib.util.spec_from_file_location(
        "trunk", ROOT / "scripts" / "train" / "train_decentralized_rl.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _scenes(seed, data, args):
    from build_operating_point_dataset import operating_point_regime
    from marl_topology.training.dynamic_frames import sample_dynamic_scenes, sample_dynamic_urban_scenes
    common = dict(seed=seed * 1000 + 777, count=args.held, node_count_choices=tuple(args.dyn_nodes),
                  regime=operating_point_regime(20.0), num_frames=args.frames, dt_s=2.0,
                  speed_min_mps=15.0, speed_max_mps=30.0, reconfig=ReconfigCost(e_edge=0.05, l_edge=0.0),
                  hold_interval=4, gamma=0.95, tag="held_")
    return (sample_dynamic_urban_scenes(rsu_count=4, blocks_per_side=3, **common) if data == "urban"
            else sample_dynamic_scenes(**common))


def _anchor_topo(obs, prev):
    budgets, edges = _budgets_edges(obs["context"])
    return local_hysteresis_action(obs["ef"], obs["edge_ids"], edges, budgets, prev,
                                   keep_threshold=0.4, add_threshold=0.6)


def _oracle_topo(obs, prev, T):
    ctx = obs["context"]
    tv = getattr(ctx, "topology_variants", None) or {}
    values = tv.values() if isinstance(tv, dict) else tv
    cands = [tuple(str(e) for e in (v.edges if hasattr(v, "edges") else v)) for v in values]
    cands = cands or [tuple(obs["edge_ids"])]
    # explicit greedy: among feasible candidates pick lowest energy, else pick highest reliability
    feas_cands = []
    inf_cands = []
    for cand in cands:
        c, e, l = T._evaluate(obs, list(cand))
        (feas_cands if c >= TAU else inf_cands).append((c, e, l, cand))
    if feas_cands:
        c, e, l, cand = min(feas_cands, key=lambda r: r[1])    # lowest energy among feasible
    else:
        c, e, l, cand = max(inf_cands, key=lambda r: r[0])     # highest reliability if none feasible
    return list(cand), c, e, l


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--data", nargs="+", default=["random", "urban"])
    ap.add_argument("--held", type=int, default=12)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--dyn-nodes", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--out", default=str(ROOT / "result_save" / "success_case_analysis.json"))
    args = ap.parse_args()
    T = _load_trunk()

    report = {"scope": "anchor (== deployable MARL, Q9/Q11 paired diff 0.000) vs central myopic oracle; "
                       "per-frame true PBFT C/E/L on the current channel; TAU=0.9", "by_data": {}}
    for data in args.data:
        frames = []                                            # one record per (scene, frame)
        for seed in args.seeds:
            for scene in _scenes(seed, data, args):
                prev_a: list = []
                for t in range(scene.n_frames):
                    obs = scene.observation(t, prev_a)
                    a_topo = _anchor_topo(obs, prev_a)
                    ac, ae, al = T._evaluate(obs, list(a_topo))
                    o_topo, oc, oe, ol = _oracle_topo(obs, prev_a, T)
                    frames.append({"a": (ac, ae, al, ac >= TAU), "o": (oc, oe, ol, oc >= TAU)})
                    prev_a = list(a_topo)                       # anchor rollout uses its own prev

        n = len(frames)
        a_feas = [f for f in frames if f["a"][3]]
        o_feas = [f for f in frames if f["o"][3]]
        both = [f for f in frames if f["a"][3] and f["o"][3]]
        a_only = [f for f in frames if f["a"][3] and not f["o"][3]]
        o_only = [f for f in frames if f["o"][3] and not f["a"][3]]
        neither = [f for f in frames if not f["a"][3] and not f["o"][3]]
        row = {
            "n_frames": n,
            "success_rate": {"anchor_eq_MARL": len(a_feas) / n, "central_oracle": len(o_feas) / n},
            "frame_partition": {"both_feasible": len(both), "anchor_only": len(a_only),
                                "oracle_only": len(o_only), "neither": len(neither)},
            "on_BOTH_feasible (shared successes)": {
                "n": len(both),
                "anchor_C": round(_mean([f["a"][0] for f in both]), 4),
                "oracle_C": round(_mean([f["o"][0] for f in both]), 4),
                "anchor_E": round(_mean([f["a"][1] for f in both]), 4),
                "oracle_E": round(_mean([f["o"][1] for f in both]), 4),
                "anchor_L": round(_mean([f["a"][2] for f in both]), 4),
                "oracle_L": round(_mean([f["o"][2] for f in both]), 4),
                "oracle_energy_saving_frac": round(
                    1 - _mean([f["o"][1] for f in both]) / max(1e-9, _mean([f["a"][1] for f in both])), 4) if both else None,
                "oracle_latency_saving_frac": round(
                    1 - _mean([f["o"][2] for f in both]) / max(1e-9, _mean([f["a"][2] for f in both])), 4) if both else None},
            "on_ORACLE_only (extra successes the oracle recovers)": {
                "n": len(o_only),
                "oracle_C": round(_mean([f["o"][0] for f in o_only]), 4) if o_only else None,
                "anchor_C_on_those_frames": round(_mean([f["a"][0] for f in o_only]), 4) if o_only else None,
                "oracle_E": round(_mean([f["o"][1] for f in o_only]), 4) if o_only else None,
                "oracle_L": round(_mean([f["o"][2] for f in o_only]), 4) if o_only else None},
            "on_ANCHOR_only (anchor solves, oracle does not)": {
                "n": len(a_only),
                "anchor_C": round(_mean([f["a"][0] for f in a_only]), 4) if a_only else None},
        }
        report["by_data"][data] = row
        print(f"=== {data} ({len(args.seeds)} seeds, {n} frames) ===")
        print(f"  success rate: anchor(==MARL) {row['success_rate']['anchor_eq_MARL']:.3f}  "
              f"central-oracle {row['success_rate']['central_oracle']:.3f}")
        print(f"  partition: both={len(both)} anchor_only={len(a_only)} oracle_only={len(o_only)} neither={len(neither)}")
        b = row["on_BOTH_feasible (shared successes)"]
        print(f"  shared successes (n={b['n']}): C {b['anchor_C']} vs {b['oracle_C']} | "
              f"E {b['anchor_E']} vs {b['oracle_E']} (oracle saves {b['oracle_energy_saving_frac']}) | "
              f"L {b['anchor_L']} vs {b['oracle_L']} (oracle saves {b['oracle_latency_saving_frac']})")
        oo = row["on_ORACLE_only (extra successes the oracle recovers)"]
        print(f"  oracle-only successes (n={oo['n']}): oracle_C {oo['oracle_C']} where anchor_C was {oo['anchor_C_on_those_frames']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
