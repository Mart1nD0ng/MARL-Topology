"""D2 science gate: the Temporal Value Test (Delta_H) on REAL multi-frame mobility data.

For each moving-vehicle DynamicScene (operating-point urban v2x_37885 regime), compose the per-frame
constrained objective into a two_timescale_env episode and compute, via the EXACT DP, the temporal
value Delta_H = J_myopic - J_horizon (Spec S3.6):

  * J_myopic  = sum of the per-frame argmin-cost topology (reconfiguration-blind chasing);
  * J_horizon = exact min discounted total cost INCLUDING reconfiguration (the DP);
  * Delta_H >= 0 always; Delta_H > 0 iff holding a good-enough topology beats chasing the per-frame
    optimum -- i.e. the task genuinely needs horizon/temporal awareness.

The per-frame cost is exactly the trunk's objective: cost = -reward_of(dense). Reconfiguration cost
= (e_edge + l_edge) * |E_t triangle E_{t-1}| (Spec S3.5). We SWEEP e_edge to show the Delta_H(reconfig)
curve -- the honest characterization of how much temporal value the real channel carries and at what
switching cost it appears. Eval-only: no training, no checkpoint selection.
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

from marl_topology.training.dynamic_frames import sample_dynamic_scenes  # noqa: E402
from marl_topology.training.two_timescale_env import (  # noqa: E402
    ReconfigCost,
    TwoTimescaleTopologyEnv,
    temporal_value_test,
)
from build_operating_point_dataset import operating_point_regime  # noqa: E402


def _load_trunk():
    spec = importlib.util.spec_from_file_location(
        "trunk", ROOT / "scripts" / "train" / "train_decentralized_rl.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _candidate_topologies(dyn, T, cap):
    """A shared, frame-invariant candidate set: the canonical named topology variants (a dict
    {name: tuple_of_edge_ids} -- empty / single_best / sparse_quorum / greedy_reliability / full_graph
    / random) plus the full graph. Edges are frame-invariant, so each is valid at every frame."""
    ctx0 = dyn.context(0)
    tv = getattr(ctx0, "topology_variants", None) or {}
    values = tv.values() if isinstance(tv, dict) else tv
    variants = []
    for v in values:
        edges = v.edges if hasattr(v, "edges") else v   # tuple of edge-id strings
        variants.append(frozenset(str(e) for e in edges))
    full = frozenset(dyn.edge_ids)
    seen, out = set(), []
    for c in [full, *variants]:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key); out.append(c)
        if len(out) >= cap:
            break
    return out


def _percentiles(xs, ps=(0.0, 0.25, 0.5, 0.75, 1.0)):
    if not xs:
        return {}
    s = sorted(xs)
    out = {}
    for p in ps:
        idx = min(len(s) - 1, int(round(p * (len(s) - 1))))
        out[f"p{int(p*100)}"] = round(s[idx], 5)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=24)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--dt", type=float, default=2.0)
    ap.add_argument("--speed-min", type=float, default=15.0)
    ap.add_argument("--speed-max", type=float, default=30.0)
    ap.add_argument("--hold-interval", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--cap-candidates", type=int, default=48)
    ap.add_argument("--e-edge-sweep", type=float, nargs="+", default=[0.0, 0.01, 0.03, 0.1, 0.3])
    ap.add_argument("--seed", type=int, default=4101)
    ap.add_argument("--tx-power", type=float, default=20.0)
    ap.add_argument("--out", default=str(ROOT / "result_save" / "dynamic_temporal_value.json"))
    args = ap.parse_args()

    T = _load_trunk()
    regime = operating_point_regime(args.tx_power)
    scenes = sample_dynamic_scenes(
        seed=args.seed, count=args.count, node_count_choices=(8, 12, 16), regime=regime,
        num_frames=args.frames, dt_s=args.dt, speed_min_mps=args.speed_min, speed_max_mps=args.speed_max,
        reconfig=ReconfigCost(), hold_interval=args.hold_interval, gamma=args.gamma)

    report = {
        "scope": "Temporal Value Test (Delta_H) on real mobility multi-frame data (eval-only)",
        "regime": "operating_point urban v2x_37885 shadowing nlosv relay-3 backhaul coverage-gated",
        "config": {"count": args.count, "frames": args.frames, "dt_s": args.dt,
                   "speed_mps": [args.speed_min, args.speed_max], "hold_interval": args.hold_interval,
                   "gamma": args.gamma, "cap_candidates": args.cap_candidates, "seed": args.seed,
                   "tau": float(T.TAU), "reward_mode": "dense"},
        "per_e_edge": {},
    }

    # Precompute per-(scene, frame) cost over the shared candidate set, so the e_edge sweep is cheap.
    scene_data = []
    for dyn in scenes:
        cands = _candidate_topologies(dyn, dyn.n_frames, args.cap_candidates)
        if len(cands) < 2:
            continue
        # e_ref per scene = full-graph energy at frame 0 (fixed normalizer, consistent across frames).
        s0 = dyn.observation(0, [])
        try:
            _c, e_full, _l = T._evaluate(s0, dyn.edge_ids)
            e_ref = max(e_full, 1e-9)
        except Exception:
            e_ref = 1e-9
        # cost[t][cand_idx] = -reward_of(dense) of cand at frame t
        cost = []
        for t in range(dyn.n_frames):
            st = dyn.observation(t, [])
            row = []
            for cand in cands:
                r, _gc, _gb, _ok = T.reward_of(st, list(cand), e_ref, 1.0, 1.0, 0.1, "dense")
                row.append(-float(r))
            cost.append(row)
        scene_data.append((dyn, cands, cost))

    report["n_scenes_evaluated"] = len(scene_data)

    for e_edge in args.e_edge_sweep:
        deltas, jmy, jho, n_pos = [], [], [], 0
        for dyn, cands, cost in scene_data:
            recon = ReconfigCost(e_edge=float(e_edge), l_edge=0.0)
            frames = list(range(dyn.n_frames))

            def cost_fn(frame_idx, topology, _cost=cost, _cands=cands):
                # exact lookup: topology is one of the shared candidates (identity by frozenset)
                return _cost[frame_idx][_cands.index(frozenset(topology))]

            env = TwoTimescaleTopologyEnv(frames, cost_fn, recon,
                                          hold_interval=dyn.hold_interval, gamma=dyn.gamma)
            cpf = [list(cands) for _ in frames]
            res = temporal_value_test(env, cpf)
            deltas.append(res["delta_h"]); jmy.append(res["j_myopic"]); jho.append(res["j_horizon"])
            if res["delta_h"] > 1e-9:
                n_pos += 1
        report["per_e_edge"][str(e_edge)] = {
            "delta_h_mean": round(sum(deltas) / max(1, len(deltas)), 5),
            "delta_h_percentiles": _percentiles(deltas),
            "scenes_with_positive_delta": n_pos,
            "scenes_total": len(scene_data),
            "frac_positive": round(n_pos / max(1, len(scene_data)), 3),
            "j_myopic_mean": round(sum(jmy) / max(1, len(jmy)), 4),
            "j_horizon_mean": round(sum(jho) / max(1, len(jho)), 4),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"scenes evaluated: {report['n_scenes_evaluated']}")
    print(f"{'e_edge':>8} {'dH_mean':>9} {'frac>0':>7} {'J_myopic':>9} {'J_horizon':>9}")
    for e, d in report["per_e_edge"].items():
        print(f"{e:>8} {d['delta_h_mean']:>9.4f} {d['frac_positive']:>7.2f} "
              f"{d['j_myopic_mean']:>9.3f} {d['j_horizon_mean']:>9.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
