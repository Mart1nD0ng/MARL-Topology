"""R4 (Belief-Guided Residual PPO) — beneficial oracle-edit dataset.

Load-bearing tests (Contract v4 §3) for the teacher dataset that supplies the direction signal R3 showed is
missing: only positive-gain LOCAL edits are supervised; each record carries ΔC/ΔD/ΔE/ΔL/ΔJ; the dataset
stores the anchor + the single edit descriptor (NEVER the full oracle topology, TechSpec §11); the generator
takes train scenes only (no held); a content hash is logged.

Fails on HEAD: `oracle_edit_dataset` does not exist yet.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT / "scripts" / "train", _ROOT / "scripts" / "diagnostics"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _scenes(n=1, frames=3, seed=4):
    from build_operating_point_dataset import operating_point_regime
    from marl_topology.training.dynamic_frames import sample_dynamic_scenes
    from marl_topology.training.two_timescale_env import ReconfigCost
    return sample_dynamic_scenes(seed=seed, count=n, node_count_choices=(8,),
                                 regime=operating_point_regime(20.0), num_frames=frames, dt_s=2.0,
                                 speed_min_mps=15.0, speed_max_mps=30.0, reconfig=ReconfigCost(),
                                 hold_interval=4, gamma=0.95)


def _build(margin=0.0):
    from marl_topology.training.oracle_edit_dataset import build_edit_dataset
    import residual_ppo_train as rp3
    T = rp3._load_trunk()
    return build_edit_dataset(_scenes(1, 3), T, margin=margin)


def test_oracle_edit_dataset_contains_anchor_and_delta() -> None:
    records, metrics = _build()
    assert len(records) > 0
    r = records[0]
    assert "anchor" in r and "kind" in r and "edge" in r and "dJ" in r
    assert r["kind"] in ("add", "remove", "swap")


def test_edit_labels_include_C_D_E_L() -> None:
    records, _ = _build()
    for r in records[:20]:
        for k in ("dC", "dD_quorum", "dE", "dL", "dJ"):
            assert k in r, f"record missing {k}"


def test_only_positive_gain_edits_supervised() -> None:
    from marl_topology.training.oracle_edit_dataset import supervised_records
    records, _ = _build(margin=0.0)
    sup = supervised_records(records, margin=0.0)
    assert all(r["dJ"] > 0.0 and r["safe"] for r in sup)            # supervised = positive gain AND safe
    # nothing positive+safe is dropped
    expected = [r for r in records if r["dJ"] > 0.0 and r["safe"]]
    assert len(sup) == len(expected)


def test_dataset_artifact_hash_logged() -> None:
    records, metrics = _build()
    assert "dataset_hash" in metrics and isinstance(metrics["dataset_hash"], str) and len(metrics["dataset_hash"]) >= 8
    from marl_topology.training.oracle_edit_dataset import dataset_hash
    assert dataset_hash(records) == metrics["dataset_hash"]          # deterministic + matches


def test_teacher_does_not_use_held_for_training() -> None:
    from marl_topology.training.oracle_edit_dataset import build_edit_dataset
    params = set(inspect.signature(build_edit_dataset).parameters)
    assert not (params & {"held", "held_scenes", "val", "val_scenes", "test", "test_scenes"}), \
        "the edit-dataset teacher must take train scenes only (held never enters)"


def test_no_full_oracle_topology_stored() -> None:
    records, _ = _build()
    for r in records[:20]:
        assert "edited_topology" not in r and "topology" not in r and "oracle" not in r
        # the edit descriptor is a single edge (add/remove) or a 2-tuple (swap) -- not a full edge set
        edge = r["edge"]
        n_edges = 1 if isinstance(edge, str) else len(edge)
        assert n_edges <= 2, "a record must store a single LOCAL edit, not a full topology"
