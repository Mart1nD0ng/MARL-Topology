"""Reproduction guard for the recovered Step-3 result (the project's validated best).

Loads the frozen K-round message-passing actors + norm stats and the operating-point held-out
shards, evaluates with the DECENTRALIZED local mutual-acceptance decoder, and asserts the
held-out feasibility reproduces ~0.82 (the value in docs/URBAN_V2X_RESEARCH_LOG.md Step-3).

Skips gracefully when the recovered artifacts/shards are not present (they live under the
gitignored result_save/), so a clean clone without the data still passes; when present, this
pins the recovered trunk so it cannot silently regress.
"""

from pathlib import Path
from random import Random

import pytest

ROOT = Path(__file__).resolve().parents[2]
RECOVERED = ROOT / "recovered_artifacts" / "recovered_step3"
ARTIFACTS = RECOVERED / "_artifacts_step3.pt"
SHARDS = [RECOVERED / f"_step3_shard_{s}.pkl" for s in (3001, 3002, 3003, 3004)]

pytestmark = pytest.mark.slow

_missing = [p.name for p in [ARTIFACTS, *SHARDS] if not p.exists()]


@pytest.mark.skipif(bool(_missing), reason=f"recovered Step-3 data absent: {_missing}")
def test_recovered_step3_reproduces_decentralized_feasibility() -> None:
    import pickle

    import torch

    from marl_topology.training.decentralized_distillation import (
        build_samples,
        feasible_rate,
        load_actor_from_state,
        local_mutual_assemble,
    )
    from marl_topology.training.production_mappo_adapter import Stage33ProductionMappoAdapter

    adapter = Stage33ProductionMappoAdapter()
    pool = []
    for shard in SHARDS:
        with open(shard, "rb") as handle:
            dataset = pickle.load(handle)
        labels = dataset.source_dataset.teacher_labels
        for split in ("train", "eval", "test"):
            for row, context in adapter.build_row_contexts(dataset, split):
                pool.append((row, context, labels[context.fixture.fixture_id]))
    Random(7).shuffle(pool)
    held = build_samples(pool[int(0.6 * len(pool)):])
    nd, ed = held[0]["nf"].shape[1], held[0]["ef"].shape[1]
    assert (nd, ed) == (8, 8)

    art = torch.load(str(ARTIFACTS), map_location="cpu", weights_only=False)
    dec = []
    for a in art["actors"]:
        actor = load_actor_from_state(a["state"], nd, ed, hidden=64, rounds=4)
        d = feasible_rate(actor, held, a["mean"], a["std"], local_mutual_assemble)
        # decentralized must match the centralized-decode ablation at the operating point.
        g = feasible_rate(actor, held, a["mean"], a["std"], __import__(
            "marl_topology.training.decentralized_distillation", fromlist=["global_argsort_assemble"]
        ).global_argsort_assemble)
        assert abs(d - g) < 1e-9, f"decentralization cost should be 0 at the operating point: {d} vs {g}"
        dec.append(d)
    mean_dec = sum(dec) / len(dec)
    assert mean_dec >= 0.81, f"recovered Step-3 decentralized feasibility regressed: {mean_dec:.4f}"
