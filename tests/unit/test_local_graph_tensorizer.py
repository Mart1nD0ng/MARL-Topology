"""Tests for tensorize_actor_graph -> LocalKHopGNNEdgeScorer end-to-end on real rows.

Confirms a full-scene set of per-node actor observations reconstructs a valid directed
candidate graph (node table + edge_index) purely from actor-safe fields, and that the
K-hop actor scores it with logits aligned to directed-edge records.
"""

import pytest

from marl_topology.models import (
    LocalGraphTensorBatch,
    LocalKHopGNNEdgeScorer,
    LocalKHopGNNEdgeScorerConfig,
    tensorize_actor_graph,
)
from marl_topology.models.tensorizers import TensorizerViolation
from marl_topology.policies.actor_interface import ActorPolicyInput


def _neighbor(neighbor_id, neighbor_kind, edge_id, distance, psucc, latency, energy):
    return {
        "neighbor_id": neighbor_id,
        "neighbor_kind": neighbor_kind,
        "edge_id": edge_id,
        "distance_3d_m": distance,
        "link_success_probability": psucc,
        "estimated_link_latency_s": latency,
        "estimated_link_energy_j": energy,
    }


def _triangle_scene():
    # Nodes v0, v1 (vehicles) and rsu0 (rsu); fully connected triangle.
    v0 = ActorPolicyInput(
        agent_id="v0",
        agent_kind="vehicle",
        time_step=0,
        local_position_m=(0.0, 0.0, 1.5),
        local_neighbor_observations=(
            _neighbor("v1", "vehicle", "v0--v1", 10.0, 0.8, 0.001, 0.01),
            _neighbor("rsu0", "rsu", "rsu0--v0", 7.0, 0.9, 0.001, 0.02),
        ),
    )
    v1 = ActorPolicyInput(
        agent_id="v1",
        agent_kind="vehicle",
        time_step=0,
        local_position_m=(10.0, 0.0, 1.5),
        local_neighbor_observations=(
            _neighbor("v0", "vehicle", "v0--v1", 10.0, 0.8, 0.001, 0.01),
            _neighbor("rsu0", "rsu", "rsu0--v1", 8.0, 0.85, 0.001, 0.02),
        ),
    )
    rsu0 = ActorPolicyInput(
        agent_id="rsu0",
        agent_kind="rsu",
        time_step=0,
        local_position_m=(5.0, 5.0, 10.0),
        local_neighbor_observations=(
            _neighbor("v0", "vehicle", "rsu0--v0", 7.0, 0.9, 0.001, 0.02),
            _neighbor("v1", "vehicle", "rsu0--v1", 8.0, 0.85, 0.001, 0.02),
        ),
    )
    return [v0, v1, rsu0]


def test_tensorize_actor_graph_builds_valid_directed_graph() -> None:
    batch = tensorize_actor_graph([_triangle_scene()])
    assert isinstance(batch, LocalGraphTensorBatch)
    assert batch.node_count == 3
    assert batch.edge_count == 6  # 3 undirected edges x 2 directions
    assert set(batch.node_ids) == {"v0", "v1", "rsu0"}
    # Every edge_index entry references a valid node, and both directions exist.
    for row in range(batch.edge_count):
        src = int(batch.edge_index[row, 0])
        dst = int(batch.edge_index[row, 1])
        assert 0 <= src < 3 and 0 <= dst < 3 and src != dst
    # Records align with edges and carry the owning agent.
    assert len(batch.records) == 6
    owners = {ref.agent_id for ref in batch.records}
    assert owners == {"v0", "v1", "rsu0"}


def test_khop_actor_scores_tensorized_graph() -> None:
    batch = tensorize_actor_graph([_triangle_scene()])
    model = LocalKHopGNNEdgeScorer(LocalKHopGNNEdgeScorerConfig(rounds=3))
    model.eval()
    logits = model.score_graph_batch(batch)
    assert logits.shape == (batch.edge_count,)


def test_tensorize_actor_graph_batches_multiple_scenes_without_cross_links() -> None:
    batch = tensorize_actor_graph([_triangle_scene(), _triangle_scene()])
    assert batch.node_count == 6
    assert batch.edge_count == 12
    # Scene 1 nodes are indices 0..2, scene 2 nodes 3..5: no edge crosses the boundary.
    for row in range(batch.edge_count):
        src = int(batch.edge_index[row, 0])
        dst = int(batch.edge_index[row, 1])
        assert (src < 3) == (dst < 3), "edge crosses scene boundary"
    assert set(batch.node_batch.tolist()) == {0, 1}


def test_tensorize_actor_graph_rejects_neighbour_without_ego() -> None:
    scene = _triangle_scene()
    # Drop rsu0's ego observation but keep it referenced as a neighbour.
    scene = [policy for policy in scene if policy.agent_id != "rsu0"]
    with pytest.raises(TensorizerViolation):
        tensorize_actor_graph([scene])
