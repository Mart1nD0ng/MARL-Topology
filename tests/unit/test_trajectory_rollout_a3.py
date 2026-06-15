"""A3 rollout evolution: build_frame_contexts produces a per-scenario sequence of
(row, context) -- one per rollout step -- on the scene advanced in time, so the env
(and the actor input) genuinely evolve across the window. The collector/loss opt-in
(frame_contexts default None) is byte-identical to the static path, covered elsewhere.
"""

from marl_topology.data.stage33_graph_structure_dataset import (
    Stage33GraphStructureConfig,
    build_stage33_graph_structure_dataset,
)
from marl_topology.scenario.scene import NodeKind
from marl_topology.training.production_mappo_adapter import Stage33ProductionMappoAdapter


def _dataset():
    return build_stage33_graph_structure_dataset(
        Stage33GraphStructureConfig(scenario_count=7, node_count_choices=(6,))
    )


def test_build_frame_contexts_aligns_with_row_contexts_and_evolves() -> None:
    dataset = _dataset()
    adapter = Stage33ProductionMappoAdapter()
    num_frames = 4
    frame_seqs = adapter.build_frame_contexts(
        dataset, "train", num_frames=num_frames, dt_s=1.0, speed_min_mps=8.0, speed_max_mps=15.0
    )
    row_contexts = adapter.build_row_contexts(dataset, "train")

    # one frame sequence per scenario slot, aligned with build_row_contexts order
    assert len(frame_seqs) == len(row_contexts)
    for seq in frame_seqs:
        assert len(seq) == num_frames
        assert all(len(pair) == 2 for pair in seq)

    seq0 = frame_seqs[0]
    static_id = row_contexts[0][1].fixture.fixture_id
    # frame fixtures are per-time and share the static base id (so row_index/step_index
    # indexing in the collector + loss resolve to the same scenario slot)
    assert seq0[0][1].fixture.fixture_id == f"{static_id}:t0"
    assert seq0[-1][1].fixture.fixture_id == f"{static_id}:t{num_frames - 1}"

    # geometry evolves: at least one vehicle moved between the first and last frame
    first_scene = seq0[0][1].fixture.scene
    last_scene = seq0[-1][1].fixture.scene
    moved = any(
        first_scene.get_node(nid).position != last_scene.get_node(nid).position
        for nid in first_scene.node_ids
        if first_scene.get_node(nid).kind is NodeKind.VEHICLE
    )
    assert moved

    # the ACTOR INPUT evolves: the actor-safe view (per-edge link metrics) changes as
    # links change with motion -- the signal a temporal actor will exploit.
    assert seq0[0][0].actor_safe_view != seq0[-1][0].actor_safe_view
    # every frame is a valid, scored context
    assert all(row.actor_safe_view for row, _ctx in seq0)


def test_build_frame_contexts_is_deterministic() -> None:
    dataset = _dataset()
    adapter = Stage33ProductionMappoAdapter()
    kwargs = dict(num_frames=3, dt_s=1.0, speed_min_mps=8.0, speed_max_mps=15.0)
    a = adapter.build_frame_contexts(dataset, "train", **kwargs)
    b = adapter.build_frame_contexts(dataset, "train", **kwargs)
    # same motions (seeded on slot index) -> identical evolved geometry
    a_pos = [f[1].fixture.scene.get_node(f[1].fixture.scene.node_ids[-1]).position for seq in a for f in seq]
    b_pos = [f[1].fixture.scene.get_node(f[1].fixture.scene.node_ids[-1]).position for seq in b for f in seq]
    assert a_pos == b_pos
