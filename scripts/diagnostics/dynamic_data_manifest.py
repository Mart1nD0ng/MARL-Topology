"""Dynamic multi-frame dataset MANIFEST (data provenance for the dynamic headline).

The dynamic 'dataset' is generated DETERMINISTICALLY from a seed (no pickled shards): the trunk's
--dynamic arm calls sample_dynamic_scenes(seed*1000+1) for train and (seed*1000+777) for held. This
manifest records, per seed, the full generator config + environment-math version + per-split scenario
ids + node-count / trajectory-length / solvability distributions + a SHA256 content hash of the frame
geometries (so a re-generation is byte-verifiable). Run after a headline to attach data provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from marl_topology.training.dynamic_frames import sample_dynamic_scenes  # noqa: E402
from marl_topology.training.two_timescale_env import ReconfigCost  # noqa: E402
from build_operating_point_dataset import operating_point_regime  # noqa: E402


def _scene_hash(dyn) -> str:
    h = hashlib.sha256()
    for sc in dyn.scenes:
        for nd in sc.nodes:
            p = nd.position
            h.update(f"{nd.node_id}:{nd.kind}:{p.x_m:.4f},{p.y_m:.4f},{p.z_m:.4f}|".encode())
    return h.hexdigest()[:16]


def _split_manifest(scenes, label):
    node_counts = Counter()
    traj_lengths = Counter()
    solv = Counter()
    ids = []
    content = hashlib.sha256()
    for dyn in scenes:
        n = len(dyn.context(0).graph.node_ids)
        node_counts[n] += 1
        traj_lengths[dyn.n_frames] += 1
        # solvability family from the scenario id suffix (feasible_sparse / infeasible / near_threshold)
        sid = dyn.sequence_id
        fam = sid.split("_")[-1] if "_" in sid else "unknown"
        solv[fam] += 1
        ids.append(sid)
        content.update(_scene_hash(dyn).encode())
    return {
        "split": label, "n_scenes": len(scenes), "scenario_ids": ids,
        "node_count_distribution": dict(sorted(node_counts.items())),
        "trajectory_length_distribution": dict(sorted(traj_lengths.items())),
        "solvability_family_distribution": dict(solv),
        "content_sha256_16": content.hexdigest()[:16],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--dyn-train", type=int, default=16)
    ap.add_argument("--dyn-held", type=int, default=16)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--dt", type=float, default=2.0)
    ap.add_argument("--speed-min", type=float, default=15.0)
    ap.add_argument("--speed-max", type=float, default=30.0)
    ap.add_argument("--hold-interval", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--reconfig-e", type=float, default=0.1)
    ap.add_argument("--dyn-nodes", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--tx-power", type=float, default=20.0)
    ap.add_argument("--out", default=str(ROOT / "result_save" / "dynamic_data_manifest.json"))
    args = ap.parse_args()

    regime = operating_point_regime(args.tx_power)
    recon = ReconfigCost(e_edge=args.reconfig_e, l_edge=0.0)
    manifest = {
        "dataset": "two_timescale_mobility_frames_v1 (deterministic by seed; no pickled shards)",
        "environment_math_version": "v2-corrected-2026-06-23 (fixed_set + one_hop_relay + "
                                    "timeout_aware_latency, relay_hops=3, tau=0.9)",
        "generator": {
            "regime": "operating_point urban v2x_37885 shadowing nlosv relay-3 backhaul coverage-gated",
            "tx_power_dbm": args.tx_power, "node_count_choices": args.dyn_nodes,
            "num_frames": args.frames, "dt_s": args.dt,
            "mobility_speed_mps": [args.speed_min, args.speed_max],
            "hold_interval": args.hold_interval, "gamma": args.gamma,
            "reconfig_e_edge": args.reconfig_e,
            "train_seed_formula": "seed*1000+1", "held_seed_formula": "seed*1000+777",
            "sampler": "sample_dynamic_scenes (advance_scene geometry; per-frame Stage21 channel; "
                       "NO per-frame SA -- feasibility read live from the evaluator)",
        },
        "per_seed": {},
    }
    for seed in args.seeds:
        train = sample_dynamic_scenes(seed=seed * 1000 + 1, count=args.dyn_train,
                                      node_count_choices=tuple(args.dyn_nodes), regime=regime,
                                      num_frames=args.frames, dt_s=args.dt, speed_min_mps=args.speed_min,
                                      speed_max_mps=args.speed_max, reconfig=recon,
                                      hold_interval=args.hold_interval, gamma=args.gamma)
        held = sample_dynamic_scenes(seed=seed * 1000 + 777, count=args.dyn_held,
                                     node_count_choices=tuple(args.dyn_nodes), regime=regime,
                                     num_frames=args.frames, dt_s=args.dt, speed_min_mps=args.speed_min,
                                     speed_max_mps=args.speed_max, reconfig=recon,
                                     hold_interval=args.hold_interval, gamma=args.gamma)
        manifest["per_seed"][str(seed)] = {
            "train": _split_manifest(train, "train"),
            "held": _split_manifest(held, "held"),
        }
    Path(args.out).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    for seed in args.seeds:
        d = manifest["per_seed"][str(seed)]
        print(f"seed {seed}: train N-dist {d['train']['node_count_distribution']} "
              f"solv {d['train']['solvability_family_distribution']} hash {d['train']['content_sha256_16']}; "
              f"held N-dist {d['held']['node_count_distribution']} hash {d['held']['content_sha256_16']}")


if __name__ == "__main__":
    main()
