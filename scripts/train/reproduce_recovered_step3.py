"""Reproduce the project's frozen best result (Step-3, held-out decentralized feasibility ~0.82
under the full TR 37.885 stochastic stack at the 4-RSU / 20 dBm operating point).

Loads the recovered frozen artifacts (3 trained K-round message-passing actors + norm stats),
reconstructs the deterministic 60/40 held-out split, and evaluates with the DECENTRALIZED
local mutual-acceptance decoder. This is the recovered, src-only reproduction of
``docs/URBAN_V2X_RESEARCH_LOG.md`` Step-3 (the logs research path was recovered into src).

Usage::

    python scripts/train/reproduce_recovered_step3.py \
        --artifacts result_save/recovered_step3/_artifacts_step3.pt \
        --shards result_save/recovered_step3/_step3_shard_3001.pkl ... 3004.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from random import Random
from statistics import fmean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pickle  # noqa: E402

import torch  # noqa: E402

from marl_topology.training.decentralized_distillation import (  # noqa: E402
    build_samples,
    ci95,
    feasible_rate,
    feasible_rate_by_n,
    global_argsort_assemble,
    load_actor_from_state,
    local_mutual_assemble,
)
from marl_topology.training.production_mappo_adapter import (  # noqa: E402
    Stage33ProductionMappoAdapter,
)

DEFAULT_DIR = ROOT / "result_save" / "recovered_step3"


def load_shard_pool(shard_paths):
    """Pool Stage-33 dataset shards into (row, context, teacher_label) tuples (driver-side I/O)."""

    adapter = Stage33ProductionMappoAdapter()
    pool = []
    for shard_path in shard_paths:
        with open(shard_path, "rb") as handle:
            dataset = pickle.load(handle)
        labels = dataset.source_dataset.teacher_labels
        for split in ("train", "eval", "test"):
            for row, context in adapter.build_row_contexts(dataset, split):
                pool.append((row, context, labels[context.fixture.fixture_id]))
    return pool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", default=str(DEFAULT_DIR / "_artifacts_step3.pt"))
    p.add_argument("--shards", nargs="+",
                   default=[str(DEFAULT_DIR / f"_step3_shard_{s}.pkl") for s in (3001, 3002, 3003, 3004)])
    p.add_argument("--held-frac", type=float, default=0.4, help="held-out fraction (Step-3 used 0.4)")
    p.add_argument("--split-seed", type=int, default=7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pool = load_shard_pool(args.shards)
    Random(args.split_seed).shuffle(pool)
    cut = int((1.0 - args.held_frac) * len(pool))
    held_items = pool[cut:]
    held = build_samples(held_items)
    ceiling = fmean(float(bool(l["feasible_exists"])) for _r, _c, l in held_items)
    nd, ed = held[0]["nf"].shape[1], held[0]["ef"].shape[1]
    art = torch.load(args.artifacts, map_location="cpu", weights_only=False)
    print(f"[repro] held={len(held)} ceiling={ceiling:.3f} node_dim={nd} edge_dim={ed} "
          f"actors={len(art['actors'])}")
    dec = []
    for i, a in enumerate(art["actors"]):
        actor = load_actor_from_state(a["state"], nd, ed, hidden=64, rounds=4)
        d, by_n = feasible_rate_by_n(actor, held, a["mean"], a["std"], local_mutual_assemble)
        g = feasible_rate(actor, held, a["mean"], a["std"], global_argsort_assemble)
        dec.append(d)
        print(f"[repro] actor {i}: DECENTRALIZED={d:.4f} (by N {by_n}) global-ablation={g:.4f}")
    m, h = ci95(dec)
    print(f"[repro] MEAN DECENTRALIZED held-out = {m:.4f} +/- {h:.4f} (teacher ceiling {ceiling:.3f})")


if __name__ == "__main__":
    main()
