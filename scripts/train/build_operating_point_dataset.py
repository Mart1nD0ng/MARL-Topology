"""Build the operating-point dataset (the validated 4-RSU / 20 dBm regime, FINAL physics).

Recovered from logs/step3_build_dataset.py. Generates Stage-33 graph-structure dataset shards at
the selected operating point: mixed N {8,12,16}, urban multi-RSU (4 RSU) with wired backhaul,
TR 37.885 V2X physics WITH stochastic shadowing + NLOSv, scheduled MAC, relay-3, and
coverage-gated PBFT membership (the frozen metric). Labels are single-realization (realization 0)
— the scene includes its realized large-scale fading, what a deployed twin would measure.

These shards feed `scripts/train/train_recovered_decentralized.py`. Building is CPU-heavy (SA
teacher + scheduled MAC + relay + stochastic channel per scene), so shard it across seeds.

Usage::

    python scripts/train/build_operating_point_dataset.py \
        --seeds 3001 3002 3003 3004 --count 60 --rsu-count 4 --tx-power 20 \
        --out-dir result_save/operating_point
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_topology.data.stage31_scenario_generator import PhysicsRegime  # noqa: E402
from marl_topology.data.stage33_graph_structure_dataset import (  # noqa: E402
    Stage33GraphStructureConfig,
    build_stage33_graph_structure_dataset,
)


def operating_point_regime(tx_power_dbm: float) -> PhysicsRegime:
    """The validated FINAL-physics operating-point regime (URBAN_V2X_RESEARCH_LOG Step-3)."""

    return PhysicsRegime(
        tx_power_dbm=tx_power_dbm,
        use_background_interference=True, orthogonal_resources=False,
        scheduled_mac=True, relay_hops=3, target_reliability=None,
        path_loss_model="v2x_37885", wired_rsu_backhaul=True,
        coverage_gated_membership=True, shadowing_37885=True, nlosv_37885=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[3001, 3002, 3003, 3004])
    p.add_argument("--count", type=int, default=60, help="scenes per shard")
    p.add_argument("--node-choices", type=int, nargs="+", default=[8, 12, 16])
    p.add_argument("--rsu-count", type=int, default=4)
    p.add_argument("--tx-power", type=float, default=20.0)
    p.add_argument("--blocks-per-side", type=int, default=3)
    p.add_argument("--out-dir", default=str(ROOT / "result_save" / "operating_point"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    regime = operating_point_regime(args.tx_power)
    print(f"[op] regime: 4-RSU operating point — tx={args.tx_power}dBm rsu={args.rsu_count} "
          f"relay=3 v2x_37885 backhaul shadowing nlosv coverage-gated; N={args.node_choices}", flush=True)
    for seed in args.seeds:
        t0 = time.time()
        out = out_dir / f"_op_shard_{seed}.pkl"
        ds = build_stage33_graph_structure_dataset(Stage33GraphStructureConfig(
            seed=seed, scenario_count=args.count, node_count_choices=tuple(args.node_choices),
            urban_mode=True, urban_blocks_per_side=args.blocks_per_side, urban_rsu_count=args.rsu_count,
            regime=regime,
            target_feasible_fraction=0.6, target_near_threshold_fraction=0.1, target_infeasible_fraction=0.3,
        ))
        with open(out, "wb") as handle:
            pickle.dump(ds, handle)
        labels = ds.source_dataset.teacher_labels
        feas = sum(int(bool(l["feasible_exists"])) for l in labels.values())
        print(f"[op] shard {seed}: {args.count} scenes in {time.time() - t0:.0f}s, "
              f"teacher-feasible {feas}/{args.count} -> {out}", flush=True)


if __name__ == "__main__":
    main()
