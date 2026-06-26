"""Build the operating-point dataset (the validated 4-RSU / 20 dBm regime, FINAL physics).

Recovered from logs/step3_build_dataset.py. Generates Stage-33 graph-structure dataset shards at
the selected operating point: mixed N {8,12,16}, urban multi-RSU (4 RSU) with wired backhaul,
TR 37.885 V2X physics WITH stochastic shadowing + NLOSv, scheduled MAC, relay-3, and
coverage-gated PBFT membership (the frozen metric). Labels are single-realization (realization 0)
— the scene includes its realized large-scale fading, what a deployed twin would measure.

These shards feed `scripts/train/train_recovered_decentralized.py`. Building is CPU-heavy (SA
teacher + scheduled MAC + relay + stochastic channel per scene) and dominates campaign wall-clock,
so each shard (one seed) builds in its own process via a spawn pool: pass ``--jobs N`` to fan the
shards across N cores. Each worker is pinned to a single BLAS/OMP thread so N workers don't
oversubscribe the machine. The build has no internal parallelism, so the outer pool is safe.

Usage::

    # 24 shards x 10 scenes built across 24 cores (240 scenes, ~minutes instead of hours)
    python scripts/train/build_operating_point_dataset.py \
        --seeds $(seq 3001 3024) --count 10 --rsu-count 4 --tx-power 20 \
        --jobs 24 --out-dir result_save/operating_point
"""

from __future__ import annotations

import os

# Pin BLAS/OMP to one thread PER PROCESS so the outer shard pool (one worker per core) does not
# oversubscribe. Set before numpy/torch import so it takes effect; setdefault lets a caller override.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

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
        # Recalibration (P0-P4 corrected env-math activated for the production dataset):
        # fixed Byzantine fault set (Spec S4.7) + one-hop relay (Spec S4.2). relay_hops=3
        # (>=2) keeps multi-hop reachability; measured to preserve feasibility (~0.731 vs
        # 0.733) -- see docs/CURRENT_HEAD_STATUS.md S10.
        fault_model="fixed_set", one_hop_relay=True, timeout_aware_latency=True,
        # D4: phase-specific PBFT message-plan accounting (Spec S4.8) -- pre_prepare = primary star,
        # prepare/commit = validator vote, clients never vote (replaces the all-pairs-x3 energy reuse).
        phase_specific_accounting=True,
    )


def _worker_init() -> None:
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def _build_one_shard(task):
    """Build a single shard (one seed) and pickle it. Module-level + picklable args for spawn."""

    (seed, count, node_choices, rsu_count, tx_power, blocks, feas, near, infeas,
     vectorized, max_total_attempts, out_dir) = task
    t0 = time.time()
    ds = build_stage33_graph_structure_dataset(Stage33GraphStructureConfig(
        seed=seed, scenario_count=count, node_count_choices=tuple(node_choices),
        urban_mode=True, urban_blocks_per_side=blocks, urban_rsu_count=rsu_count,
        regime=operating_point_regime(tx_power),
        target_feasible_fraction=feas, target_near_threshold_fraction=near,
        target_infeasible_fraction=infeas,
        vectorized_evaluator=vectorized, max_total_attempts=max_total_attempts,
    ))
    out = Path(out_dir) / f"_op_shard_{seed}.pkl"
    with open(out, "wb") as handle:
        pickle.dump(ds, handle)
    labels = ds.source_dataset.teacher_labels
    feas_n = sum(int(bool(l["feasible_exists"])) for l in labels.values())
    return seed, count, feas_n, str(out), time.time() - t0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[3001, 3002, 3003, 3004])
    p.add_argument("--count", type=int, default=60, help="scenes per shard")
    p.add_argument("--node-choices", type=int, nargs="+", default=[8, 12, 16])
    p.add_argument("--rsu-count", type=int, default=4)
    p.add_argument("--tx-power", type=float, default=20.0)
    p.add_argument("--blocks-per-side", type=int, default=3)
    # Scene-family split. Defaults reproduce the validated operating-point dataset (0.6/0.1/0.3).
    # A denser/higher-power deployment naturally supplies more feasible scenes, so a richer
    # regime can raise --feasible-frac (e.g. 0.8/0.1/0.1) instead of thrashing to fill a 30%
    # infeasible bin the distribution no longer produces. Must sum to 1.0.
    p.add_argument("--feasible-frac", type=float, default=0.6)
    p.add_argument("--near-frac", type=float, default=0.1)
    p.add_argument("--infeasible-frac", type=float, default=0.3)
    p.add_argument("--vectorized", action="store_true",
                   help="route the SA/measure path through the bounded-cache 6x-faster "
                        "VectorizedStage21Evaluator (float-identical) -- REQUIRED for N>=24 builds")
    p.add_argument("--max-total-attempts", type=int, default=0,
                   help="hard cap on the rejection-sampling loop (0=auto=count*40); bounds low-yield "
                        "N>=24 builds so they return a smaller dataset instead of thrashing for hours")
    p.add_argument("--jobs", type=int, default=0,
                   help="parallel shard builds (0 = min(num seeds, cpu_count-1))")
    p.add_argument("--out-dir", default=str(ROOT / "result_save" / "operating_point"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = args.jobs or min(len(args.seeds), max(1, (os.cpu_count() or 2) - 1))
    print(f"[op] regime: tx={args.tx_power}dBm rsu={args.rsu_count} blocks={args.blocks_per_side} "
          f"relay=3 v2x_37885 backhaul shadowing nlosv coverage-gated; N={args.node_choices}; "
          f"split feas/near/infeas={args.feasible_frac}/{args.near_frac}/{args.infeasible_frac}; "
          f"{len(args.seeds)} shards x {args.count} scenes, jobs={jobs}", flush=True)
    tasks = [(seed, args.count, args.node_choices, args.rsu_count, args.tx_power,
              args.blocks_per_side, args.feasible_frac, args.near_frac, args.infeasible_frac,
              args.vectorized, args.max_total_attempts, str(out_dir)) for seed in args.seeds]

    def _report(result):
        seed, count, feas_n, out, dt = result
        print(f"[op] shard {seed}: {count} scenes in {dt:.0f}s, "
              f"teacher-feasible {feas_n}/{count} -> {out}", flush=True)

    if jobs <= 1 or len(tasks) == 1:
        _worker_init()
        for task in tasks:
            _report(_build_one_shard(task))
    else:
        # spawn (not fork) + single-thread workers: the known-good Linux pattern for this stack.
        with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("spawn"),
                                 initializer=_worker_init) as pool:
            for result in pool.map(_build_one_shard, tasks):
                _report(result)


if __name__ == "__main__":
    main()
