"""A4: the leakage-safe, read-only temporal history window [E, W, F] for the GRU actor.

Built ONLY from already-validated actor edge tensors of past+present A3 frames, aligned
to the current frame's edges. It never touches actor_safe_view (so the actor-safe
boundary is untouched); the static path produces no window (covered by the full suite).
"""

import torch

from marl_topology.data.stage33_graph_structure_dataset import (
    Stage33GraphStructureConfig,
    build_stage33_graph_structure_dataset,
)
from marl_topology.models.tensorizers import (
    tensorize_actor_history_sequence,
    tensorize_actor_policy_inputs,
)
from marl_topology.training.policy_gradient.pilot_runner import (
    _policy_input_from_stage22_actor_row,
)
from marl_topology.training.production_mappo_adapter import Stage33ProductionMappoAdapter


def _frame_inputs(row):
    return [_policy_input_from_stage22_actor_row(r) for r in row.actor_safe_view]


def _frame_seq():
    dataset = build_stage33_graph_structure_dataset(
        Stage33GraphStructureConfig(scenario_count=7, node_count_choices=(6,))
    )
    adapter = Stage33ProductionMappoAdapter()
    return adapter.build_frame_contexts(
        dataset, "train", num_frames=4, dt_s=1.0, speed_min_mps=8.0, speed_max_mps=15.0
    )[0]


def test_history_window_shape_alignment_and_evolution() -> None:
    seq = _frame_seq()
    window = len(seq)  # full window at the last step
    frames_inputs = [_frame_inputs(row) for row, _ctx in seq]  # oldest..current
    history = tensorize_actor_history_sequence(frames_inputs, window=window)

    current = tensorize_actor_policy_inputs(frames_inputs[-1])
    edge_count, feature_dim = current.edge_features.shape
    assert history.shape == (edge_count, window, feature_dim)

    # the present (last) time slice equals the live current-frame edge tensor (row axis
    # aligned by directed_edge_id), so history pairs 1:1 with what the GNN scores.
    assert torch.allclose(history[:, -1, :], current.edge_features)

    # the history EVOLVES: the oldest frame differs from the present (vehicles moved).
    assert not torch.allclose(history[:, 0, :], history[:, -1, :])


def test_history_window_left_pads_at_early_step() -> None:
    seq = _frame_seq()
    window = 4
    # the window for the FIRST step (only frame 0 is past+present) is left-padded.
    history = tensorize_actor_history_sequence([_frame_inputs(seq[0][0])], window=window)
    current = tensorize_actor_policy_inputs(_frame_inputs(seq[0][0]))
    assert history.shape == (current.edge_features.shape[0], window, current.edge_features.shape[1])
    # only the present (last) slice is filled; the W-1 leading pad slices are zero.
    assert torch.count_nonzero(history[:, :-1, :]) == 0
    assert torch.count_nonzero(history[:, -1, :]) > 0
    assert torch.allclose(history[:, -1, :], current.edge_features)


def test_history_window_rejects_bad_window() -> None:
    seq = _frame_seq()
    import pytest

    with pytest.raises(ValueError):
        tensorize_actor_history_sequence([_frame_inputs(seq[0][0])], window=0)
