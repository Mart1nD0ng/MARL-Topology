"""Tensorization utilities for actor-local edge scoring and critic baselines."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

import torch

from marl_topology.policies.actor_interface import ActorPolicyInput
from marl_topology.policies.edge_scores import (
    EdgeScoreBatch,
    EdgeScoreRecord,
    make_directed_edge_id,
)
from marl_topology.policies.interface_contract import (
    ACTOR_POLICY_FORBIDDEN_INPUT_FIELDS,
    validate_actor_policy_input_row,
)


ACTOR_EDGE_FEATURE_SCHEMA_ID = "actor_local_edge_tensor_v1"
ACTOR_EDGE_FEATURE_FIELDS = (
    "agent_kind_is_vehicle",
    "agent_kind_is_rsu",
    "neighbor_kind_is_vehicle",
    "neighbor_kind_is_rsu",
    "local_position_x_m",
    "local_position_y_m",
    "local_position_z_m",
    "distance_3d_m",
    "link_success_probability",
    "estimated_link_latency_s",
    "estimated_link_energy_j",
)
# Stage 31 (B3): a constraint-aware actor feature schema. It extends v1 with two
# actor-safe, purely local signals of endpoint contention so the actor is no
# longer blind to the endpoint-budget constraint the assembler enforces. Both
# fields are derived from the agent's own local neighbor observations (its local
# incident-edge degree); no global, budget-table, or oracle field is introduced.
ACTOR_EDGE_FEATURE_SCHEMA_ID_V2 = "actor_local_edge_tensor_v2_budget_aware"
ACTOR_EDGE_CONSTRAINT_FIELDS = (
    "local_incident_edge_count",
    "endpoint_contention",
)
ACTOR_EDGE_FEATURE_FIELDS_V2 = ACTOR_EDGE_FEATURE_FIELDS + ACTOR_EDGE_CONSTRAINT_FIELDS

# Node-level actor-safe features for the K-hop full-graph message-passing actor. Every
# field is derived from a node's OWN ego observation (its position, kind, and local
# incident-edge degree); no global, budget-table, or oracle field is introduced. These
# are the node states the K-hop actor encodes and propagates over the candidate graph.
ACTOR_NODE_FEATURE_SCHEMA_ID = "actor_local_node_tensor_v1"
ACTOR_NODE_FEATURE_FIELDS = (
    "position_x_m",
    "position_y_m",
    "position_z_m",
    "kind_is_vehicle",
    "kind_is_rsu",
    "local_incident_edge_count",
    "endpoint_contention",
)

CRITIC_GLOBAL_FEATURE_SCHEMA_ID = "centralized_critic_global_tensor_v1"
CRITIC_GLOBAL_FEATURE_FIELDS = (
    "node_count",
    "candidate_edge_count",
    "selected_edge_count",
    "selected_edge_density",
    "time_step_mean",
)


class TensorizerViolation(ValueError):
    """Raised when model tensorization would cross a declared boundary."""


@dataclass(frozen=True, slots=True)
class ActorEdgeRecordRef:
    sample_index: int
    agent_id: str
    neighbor_id: str
    edge_id: str
    directed_edge_id: str
    time_step: int


@dataclass(frozen=True, slots=True)
class ActorEdgeTensorBatch:
    edge_features: torch.Tensor
    edge_mask: torch.Tensor
    group_ids: torch.Tensor
    records: tuple[ActorEdgeRecordRef, ...]
    feature_schema_id: str = ACTOR_EDGE_FEATURE_SCHEMA_ID
    # Optional read-only [E, W, F] temporal history (A4) aligned row-for-row with
    # edge_features, consumed by the temporal actor (Part B). Default None keeps every
    # existing construction + the static path byte-identical (the guard below is
    # skipped entirely when None).
    history: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.feature_schema_id != ACTOR_EDGE_FEATURE_SCHEMA_ID:
            raise TensorizerViolation("unexpected actor edge feature schema")
        if self.edge_features.ndim != 2:
            raise TensorizerViolation("edge_features must have shape [edge_count, feature_dim]")
        if self.edge_features.shape[1] != len(ACTOR_EDGE_FEATURE_FIELDS):
            raise TensorizerViolation("actor edge feature dimension mismatch")
        if self.edge_mask.shape != (self.edge_features.shape[0],):
            raise TensorizerViolation("edge_mask shape must match edge count")
        if self.group_ids.shape != (self.edge_features.shape[0],):
            raise TensorizerViolation("group_ids shape must match edge count")
        if len(self.records) != self.edge_features.shape[0]:
            raise TensorizerViolation("record count must match edge feature count")
        if self.history is not None:
            if self.history.ndim != 3:
                raise TensorizerViolation("history must be [edge_count, window, feature_dim]")
            if self.history.shape[0] != self.edge_features.shape[0]:
                raise TensorizerViolation("history edge count must match edge_features")
            if self.history.shape[2] != len(ACTOR_EDGE_FEATURE_FIELDS):
                raise TensorizerViolation("history feature dimension mismatch")

    @property
    def edge_count(self) -> int:
        return int(self.edge_features.shape[0])

    def to_edge_score_batch(
        self,
        logits: torch.Tensor,
        *,
        batch_id: str,
        source: str,
        include_probability: bool = True,
    ) -> EdgeScoreBatch:
        flat_logits = logits.reshape(-1).detach().cpu()
        if flat_logits.shape[0] != self.edge_count:
            raise TensorizerViolation("logit count must match tensorized edge count")
        probabilities = torch.sigmoid(flat_logits) if include_probability else None
        records = []
        for index, ref in enumerate(self.records):
            probability = (
                float(probabilities[index].item()) if probabilities is not None else None
            )
            records.append(
                EdgeScoreRecord(
                    agent_id=ref.agent_id,
                    neighbor_id=ref.neighbor_id,
                    edge_id=ref.edge_id,
                    directed_edge_id=ref.directed_edge_id,
                    score=float(flat_logits[index].item()),
                    probability=probability,
                    score_source=source,
                    time_step=ref.time_step,
                )
            )
        return EdgeScoreBatch.from_records(records, batch_id=batch_id, source=source)


@dataclass(frozen=True, slots=True)
class CriticTensorBatch:
    global_features: torch.Tensor
    edge_mask: torch.Tensor
    candidate_edge_ids: tuple[str, ...]
    feature_schema_id: str = CRITIC_GLOBAL_FEATURE_SCHEMA_ID

    def __post_init__(self) -> None:
        if self.feature_schema_id != CRITIC_GLOBAL_FEATURE_SCHEMA_ID:
            raise TensorizerViolation("unexpected critic feature schema")
        if self.global_features.ndim != 2:
            raise TensorizerViolation("critic global_features must be [batch, feature_dim]")
        if self.global_features.shape[1] != len(CRITIC_GLOBAL_FEATURE_FIELDS):
            raise TensorizerViolation("critic global feature dimension mismatch")
        expected_mask_shape = (self.global_features.shape[0], len(self.candidate_edge_ids))
        if self.edge_mask.shape != expected_mask_shape:
            raise TensorizerViolation("critic edge_mask shape mismatch")

    @property
    def batch_size(self) -> int:
        return int(self.global_features.shape[0])

    @property
    def edge_count(self) -> int:
        return len(self.candidate_edge_ids)


def tensorize_actor_policy_inputs(
    policy_inputs: Iterable[ActorPolicyInput],
) -> ActorEdgeTensorBatch:
    features: list[tuple[float, ...]] = []
    refs: list[ActorEdgeRecordRef] = []
    for sample_index, policy_input in enumerate(policy_inputs):
        validate_actor_policy_input_row(policy_input.to_actor_safe_row())
        for neighbor in policy_input.local_neighbor_observations:
            neighbor_id = str(_neighbor_value(neighbor, "neighbor_id"))
            edge_id = str(_neighbor_value(neighbor, "edge_id"))
            directed_edge_id = make_directed_edge_id(policy_input.agent_id, neighbor_id)
            feature_row = (
                _kind_flag(policy_input.agent_kind, "vehicle"),
                _kind_flag(policy_input.agent_kind, "rsu"),
                _kind_flag(str(_neighbor_value(neighbor, "neighbor_kind")), "vehicle"),
                _kind_flag(str(_neighbor_value(neighbor, "neighbor_kind")), "rsu"),
                _finite("local_position_x_m", policy_input.local_position_m[0]),
                _finite("local_position_y_m", policy_input.local_position_m[1]),
                _finite("local_position_z_m", policy_input.local_position_m[2]),
                _finite("distance_3d_m", _neighbor_value(neighbor, "distance_3d_m")),
                _finite(
                    "link_success_probability",
                    _neighbor_value(neighbor, "link_success_probability"),
                ),
                _finite(
                    "estimated_link_latency_s",
                    _neighbor_value(neighbor, "estimated_link_latency_s"),
                ),
                _finite(
                    "estimated_link_energy_j",
                    _neighbor_value(neighbor, "estimated_link_energy_j"),
                ),
            )
            features.append(feature_row)
            refs.append(
                ActorEdgeRecordRef(
                    sample_index=sample_index,
                    agent_id=policy_input.agent_id,
                    neighbor_id=neighbor_id,
                    edge_id=edge_id,
                    directed_edge_id=directed_edge_id,
                    time_step=policy_input.time_step,
                )
            )
    order = sorted(range(len(refs)), key=lambda index: refs[index].directed_edge_id)
    ordered_features = [features[index] for index in order]
    ordered_refs = tuple(refs[index] for index in order)
    feature_tensor = torch.tensor(ordered_features, dtype=torch.float32)
    if feature_tensor.numel() == 0:
        feature_tensor = torch.empty((0, len(ACTOR_EDGE_FEATURE_FIELDS)), dtype=torch.float32)
    return ActorEdgeTensorBatch(
        edge_features=feature_tensor,
        edge_mask=torch.ones((feature_tensor.shape[0],), dtype=torch.bool),
        group_ids=torch.tensor(
            [ref.sample_index for ref in ordered_refs],
            dtype=torch.long,
        ),
        records=ordered_refs,
    )


LOCAL_GRAPH_TENSOR_SCHEMA_ID = "actor_local_graph_tensor_v1"


@dataclass(frozen=True, slots=True)
class LocalGraphTensorBatch:
    """Batched candidate-graph tensors for the K-hop full-graph message-passing actor.

    One or more scenes are concatenated with disjoint node-index ranges. ``edge_index``
    holds ``(src, dst)`` GLOBAL node indices for **directed** edges (both directions of
    every physical link are present, so destination aggregation is bidirectional).
    ``records`` are the directed edge refs aligned row-for-row with ``edge_features`` and
    with the actor's output logits, so logits map back to directed edges exactly like the
    flat ``ActorEdgeTensorBatch`` path. ``node_batch`` / ``edge_batch`` label the scene of
    each node / edge so normalisation stays scene-local.
    """

    node_features: torch.Tensor
    edge_features: torch.Tensor
    edge_index: torch.Tensor
    node_batch: torch.Tensor
    edge_batch: torch.Tensor
    node_mask: torch.Tensor
    edge_mask: torch.Tensor
    records: tuple[ActorEdgeRecordRef, ...]
    node_ids: tuple[str, ...]
    node_feature_schema_id: str = ACTOR_NODE_FEATURE_SCHEMA_ID
    edge_feature_schema_id: str = ACTOR_EDGE_FEATURE_SCHEMA_ID
    schema_id: str = LOCAL_GRAPH_TENSOR_SCHEMA_ID

    def __post_init__(self) -> None:
        if self.node_feature_schema_id != ACTOR_NODE_FEATURE_SCHEMA_ID:
            raise TensorizerViolation("unexpected actor node feature schema")
        if self.edge_feature_schema_id != ACTOR_EDGE_FEATURE_SCHEMA_ID:
            raise TensorizerViolation("unexpected actor edge feature schema")
        if self.node_features.ndim != 2 or self.node_features.shape[1] != len(ACTOR_NODE_FEATURE_FIELDS):
            raise TensorizerViolation("node_features must be [node_count, node_feature_dim]")
        if self.edge_features.ndim != 2 or self.edge_features.shape[1] != len(ACTOR_EDGE_FEATURE_FIELDS):
            raise TensorizerViolation("edge_features must be [edge_count, edge_feature_dim]")
        if self.edge_index.shape != (self.edge_features.shape[0], 2):
            raise TensorizerViolation("edge_index must be [edge_count, 2]")
        if self.node_batch.shape != (self.node_features.shape[0],):
            raise TensorizerViolation("node_batch shape must match node count")
        if self.edge_batch.shape != (self.edge_features.shape[0],):
            raise TensorizerViolation("edge_batch shape must match edge count")
        if self.node_mask.shape != (self.node_features.shape[0],):
            raise TensorizerViolation("node_mask shape must match node count")
        if self.edge_mask.shape != (self.edge_features.shape[0],):
            raise TensorizerViolation("edge_mask shape must match edge count")
        if len(self.records) != self.edge_features.shape[0]:
            raise TensorizerViolation("record count must match edge feature count")
        if len(self.node_ids) != self.node_features.shape[0]:
            raise TensorizerViolation("node id count must match node feature count")

    @property
    def node_count(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def edge_count(self) -> int:
        return int(self.edge_features.shape[0])

    def to_edge_score_batch(
        self,
        logits: torch.Tensor,
        *,
        batch_id: str,
        source: str,
        include_probability: bool = True,
    ) -> EdgeScoreBatch:
        """Directed ``EdgeScoreRecord`` batch aligned to ``records`` (for the assembler)."""

        flat_logits = logits.reshape(-1).detach().cpu()
        if flat_logits.shape[0] != self.edge_count:
            raise TensorizerViolation("logit count must match graph edge count")
        probabilities = torch.sigmoid(flat_logits) if include_probability else None
        records = []
        for index, ref in enumerate(self.records):
            probability = float(probabilities[index].item()) if probabilities is not None else None
            records.append(
                EdgeScoreRecord(
                    agent_id=ref.agent_id,
                    neighbor_id=ref.neighbor_id,
                    edge_id=ref.edge_id,
                    directed_edge_id=ref.directed_edge_id,
                    score=float(flat_logits[index].item()),
                    probability=probability,
                    score_source=source,
                    time_step=ref.time_step,
                )
            )
        return EdgeScoreBatch.from_records(records, batch_id=batch_id, source=source)

    @property
    def directed_edge_ids(self) -> tuple[str, ...]:
        return tuple(ref.directed_edge_id for ref in self.records)


def tensorize_actor_graph(
    scenes: Iterable[Iterable[ActorPolicyInput]],
) -> LocalGraphTensorBatch:
    """Build a batched candidate-graph tensor from per-scene actor observations.

    Each scene is the full set of per-node ego observations (one ``ActorPolicyInput`` per
    node). Node identities and the directed ``edge_index`` are reconstructed purely from
    actor-safe fields already present in the rows (``agent_id``, ``neighbor_id``,
    ``edge_id``, ``local_position_m``, ``agent_kind``, the local neighbour observations);
    no global, oracle, or metric field is read. Each scene must observe every node it
    references as a neighbour (the full-scene contract), else a violation is raised.
    """

    node_feature_rows: list[tuple[float, ...]] = []
    node_ids: list[str] = []
    node_batch: list[int] = []
    edge_feature_rows: list[tuple[float, ...]] = []
    edge_index_rows: list[tuple[int, int]] = []
    edge_batch: list[int] = []
    refs: list[ActorEdgeRecordRef] = []
    node_offset = 0
    for scene_index, scene in enumerate(scenes):
        inputs = list(scene)
        local_index: dict[str, int] = {}
        kind_by_id: dict[str, str] = {}
        position_by_id: dict[str, tuple[float, float, float]] = {}
        degree_by_id: dict[str, int] = {}
        time_step_by_id: dict[str, int] = {}
        for policy_input in inputs:
            validate_actor_policy_input_row(policy_input.to_actor_safe_row())
            agent_id = str(policy_input.agent_id)
            if agent_id in local_index:
                raise TensorizerViolation(f"duplicate ego observation for node: {agent_id}")
            local_index[agent_id] = len(local_index)
            kind_by_id[agent_id] = str(policy_input.agent_kind)
            position_by_id[agent_id] = (
                _finite("position_x_m", policy_input.local_position_m[0]),
                _finite("position_y_m", policy_input.local_position_m[1]),
                _finite("position_z_m", policy_input.local_position_m[2]),
            )
            degree_by_id[agent_id] = len(tuple(policy_input.local_neighbor_observations))
            time_step_by_id[agent_id] = int(policy_input.time_step)
        # Node feature table (sorted by the stable ego enumeration order above).
        for agent_id, index in sorted(local_index.items(), key=lambda item: item[1]):
            degree = float(degree_by_id[agent_id])
            contention = 0.0 if degree <= 1.0 else 1.0 - 1.0 / degree
            px, py, pz = position_by_id[agent_id]
            node_feature_rows.append(
                (
                    px,
                    py,
                    pz,
                    _kind_flag(kind_by_id[agent_id], "vehicle"),
                    _kind_flag(kind_by_id[agent_id], "rsu"),
                    degree,
                    contention,
                )
            )
            node_ids.append(agent_id)
            node_batch.append(scene_index)
        # Directed edge rows.
        scene_edges: list[tuple[str, ActorEdgeRecordRef, tuple[float, ...]]] = []
        for policy_input in inputs:
            agent_id = str(policy_input.agent_id)
            for neighbor in policy_input.local_neighbor_observations:
                neighbor_id = str(_neighbor_value(neighbor, "neighbor_id"))
                if neighbor_id not in local_index:
                    raise TensorizerViolation(
                        f"neighbour {neighbor_id} of {agent_id} has no ego observation in scene"
                    )
                edge_id = str(_neighbor_value(neighbor, "edge_id"))
                directed_edge_id = make_directed_edge_id(agent_id, neighbor_id)
                feature_row = (
                    _kind_flag(kind_by_id[agent_id], "vehicle"),
                    _kind_flag(kind_by_id[agent_id], "rsu"),
                    _kind_flag(str(_neighbor_value(neighbor, "neighbor_kind")), "vehicle"),
                    _kind_flag(str(_neighbor_value(neighbor, "neighbor_kind")), "rsu"),
                    _finite("local_position_x_m", policy_input.local_position_m[0]),
                    _finite("local_position_y_m", policy_input.local_position_m[1]),
                    _finite("local_position_z_m", policy_input.local_position_m[2]),
                    _finite("distance_3d_m", _neighbor_value(neighbor, "distance_3d_m")),
                    _finite(
                        "link_success_probability",
                        _neighbor_value(neighbor, "link_success_probability"),
                    ),
                    _finite(
                        "estimated_link_latency_s",
                        _neighbor_value(neighbor, "estimated_link_latency_s"),
                    ),
                    _finite(
                        "estimated_link_energy_j",
                        _neighbor_value(neighbor, "estimated_link_energy_j"),
                    ),
                )
                ref = ActorEdgeRecordRef(
                    sample_index=scene_index,
                    agent_id=agent_id,
                    neighbor_id=neighbor_id,
                    edge_id=edge_id,
                    directed_edge_id=directed_edge_id,
                    time_step=int(policy_input.time_step),
                )
                scene_edges.append((directed_edge_id, ref, feature_row))
        scene_edges.sort(key=lambda item: item[0])
        for _directed_edge_id, ref, feature_row in scene_edges:
            edge_feature_rows.append(feature_row)
            edge_index_rows.append(
                (node_offset + local_index[ref.agent_id], node_offset + local_index[ref.neighbor_id])
            )
            edge_batch.append(scene_index)
            refs.append(ref)
        node_offset += len(local_index)

    node_count = len(node_feature_rows)
    edge_count = len(edge_feature_rows)
    node_tensor = (
        torch.tensor(node_feature_rows, dtype=torch.float32)
        if node_count
        else torch.empty((0, len(ACTOR_NODE_FEATURE_FIELDS)), dtype=torch.float32)
    )
    edge_tensor = (
        torch.tensor(edge_feature_rows, dtype=torch.float32)
        if edge_count
        else torch.empty((0, len(ACTOR_EDGE_FEATURE_FIELDS)), dtype=torch.float32)
    )
    edge_index = (
        torch.tensor(edge_index_rows, dtype=torch.long)
        if edge_count
        else torch.empty((0, 2), dtype=torch.long)
    )
    return LocalGraphTensorBatch(
        node_features=node_tensor,
        edge_features=edge_tensor,
        edge_index=edge_index,
        node_batch=torch.tensor(node_batch, dtype=torch.long),
        edge_batch=torch.tensor(edge_batch, dtype=torch.long),
        node_mask=torch.ones((node_count,), dtype=torch.bool),
        edge_mask=torch.ones((edge_count,), dtype=torch.bool),
        records=tuple(refs),
        node_ids=tuple(node_ids),
    )


ACTOR_EDGE_HISTORY_SCHEMA_ID = "actor_local_edge_history_v1"


def tensorize_actor_history_sequence(
    frames: Sequence[Iterable[ActorPolicyInput]],
    *,
    window: int,
) -> torch.Tensor:
    """Read-only ``[E, W, F]`` temporal history aligned to the CURRENT frame's edges (A4).

    ``frames`` is oldest-first, one entry per past+present time step, each a set of
    ActorPolicyInputs (the agent's OWN local view at that frame). The LAST frame is the
    current decision point; its directed-edge order (the same ``directed_edge_id`` sort
    ``tensorize_actor_policy_inputs`` produces) defines the row (E) axis, so history row
    ``i`` pairs 1:1 with the live current-frame edge tensor. Each frame is tensorized
    with the SAME ``tensorize_actor_policy_inputs`` and aligned by ``directed_edge_id``;
    edges/steps absent in a past frame (a neighbour moved out of range) are zero-padded,
    and the window is left-padded to ``W``. The GRU temporal actor (Part B) consumes
    ``[E, W, F]`` with ``F == len(ACTOR_EDGE_FEATURE_FIELDS)``.

    Leakage-safe by construction: built ONLY from already-validated actor edge tensors
    (every frame passes through ``validate_actor_policy_input_row`` inside
    ``tensorize_actor_policy_inputs``), strictly from the frames the caller passes (which
    are only past+present -- never future), and NEVER written back into any actor-safe
    row, so the actor-safe boundary and its contract tests are untouched.
    """

    if window < 1:
        raise ValueError("window must be >= 1")
    batches = [tensorize_actor_policy_inputs(frame) for frame in frames]
    if not batches:
        raise ValueError("history needs at least one frame")
    current = batches[-1]
    feature_dim = int(current.edge_features.shape[1])
    anchor = [ref.directed_edge_id for ref in current.records]
    edge_count = len(anchor)
    recent = batches[-window:]
    left_pad = window - len(recent)
    history = torch.zeros((edge_count, window, feature_dim), dtype=torch.float32)
    for offset, batch in enumerate(recent):
        time_index = left_pad + offset
        by_edge = {
            ref.directed_edge_id: batch.edge_features[row]
            for row, ref in enumerate(batch.records)
        }
        for edge_index, directed_edge_id in enumerate(anchor):
            feature = by_edge.get(directed_edge_id)
            if feature is not None:
                history[edge_index, time_index, :] = feature
    return history


@dataclass(frozen=True, slots=True)
class ActorEdgeTensorBatchV2:
    """Constraint-aware actor edge tensor batch (Stage 31 v2 schema)."""

    edge_features: torch.Tensor
    edge_mask: torch.Tensor
    group_ids: torch.Tensor
    records: tuple[ActorEdgeRecordRef, ...]
    feature_schema_id: str = ACTOR_EDGE_FEATURE_SCHEMA_ID_V2

    def __post_init__(self) -> None:
        if self.feature_schema_id != ACTOR_EDGE_FEATURE_SCHEMA_ID_V2:
            raise TensorizerViolation("unexpected actor edge v2 feature schema")
        if self.edge_features.ndim != 2:
            raise TensorizerViolation("edge_features must be [edge_count, feature_dim]")
        if self.edge_features.shape[1] != len(ACTOR_EDGE_FEATURE_FIELDS_V2):
            raise TensorizerViolation("actor edge v2 feature dimension mismatch")
        if self.edge_mask.shape != (self.edge_features.shape[0],):
            raise TensorizerViolation("edge_mask shape must match edge count")
        if self.group_ids.shape != (self.edge_features.shape[0],):
            raise TensorizerViolation("group_ids shape must match edge count")
        if len(self.records) != self.edge_features.shape[0]:
            raise TensorizerViolation("record count must match edge feature count")

    @property
    def edge_count(self) -> int:
        return int(self.edge_features.shape[0])


def tensorize_actor_policy_inputs_v2(
    policy_inputs: Iterable[ActorPolicyInput],
) -> ActorEdgeTensorBatchV2:
    """Tensorize actor inputs with the v2 constraint-aware (budget) features.

    The two extra columns are local endpoint contention signals derived only
    from the agent's own incident-edge degree, keeping the actor decentralized.
    """

    base = tensorize_actor_policy_inputs(policy_inputs)
    if base.edge_count == 0:
        empty = torch.empty((0, len(ACTOR_EDGE_FEATURE_FIELDS_V2)), dtype=torch.float32)
        return ActorEdgeTensorBatchV2(
            edge_features=empty,
            edge_mask=base.edge_mask,
            group_ids=base.group_ids,
            records=base.records,
        )
    # Local incident-edge degree per agent within this batch.
    degree_by_agent: dict[str, int] = {}
    for ref in base.records:
        degree_by_agent[ref.agent_id] = degree_by_agent.get(ref.agent_id, 0) + 1
    extra_rows = []
    for ref in base.records:
        degree = float(degree_by_agent[ref.agent_id])
        # endpoint_contention -> 0 when the endpoint hosts one edge, approaching 1
        # as the endpoint must choose among many incident edges under its budget.
        contention = 0.0 if degree <= 1.0 else 1.0 - 1.0 / degree
        extra_rows.append((degree, contention))
    extra_tensor = torch.tensor(extra_rows, dtype=torch.float32)
    combined = torch.cat([base.edge_features, extra_tensor], dim=1)
    return ActorEdgeTensorBatchV2(
        edge_features=combined,
        edge_mask=base.edge_mask,
        group_ids=base.group_ids,
        records=base.records,
    )


def tensorize_actor_policy_rows(rows: Iterable[Mapping[str, object]]) -> ActorEdgeTensorBatch:
    policy_inputs = []
    for row in rows:
        forbidden = sorted(set(row) & set(ACTOR_POLICY_FORBIDDEN_INPUT_FIELDS))
        if forbidden:
            raise TensorizerViolation(f"forbidden actor tensorizer fields: {forbidden}")
        validate_actor_policy_input_row(row)
        policy_inputs.append(ActorPolicyInput.from_actor_safe_row(row))
    return tensorize_actor_policy_inputs(policy_inputs)


def tensorize_critic_evidence_rows(rows: Iterable[Mapping[str, object]]) -> CriticTensorBatch:
    row_tuple = tuple(rows)
    if not row_tuple:
        raise TensorizerViolation("critic tensorization requires at least one evidence row")
    candidate_ids = tuple(
        sorted(
            {
                str(edge_id)
                for row in row_tuple
                for edge_id in _critic_view(row).get("candidate_edge_ids", ())
            }
        )
    )
    if not candidate_ids:
        raise TensorizerViolation("critic rows must declare candidate_edge_ids")
    candidate_set = set(candidate_ids)
    feature_rows: list[tuple[float, ...]] = []
    edge_mask_rows: list[tuple[bool, ...]] = []
    for row in row_tuple:
        view = _critic_view(row)
        row_candidates = tuple(str(edge_id) for edge_id in view.get("candidate_edge_ids", ()))
        selected_edges = tuple(str(edge_id) for edge_id in view.get("selected_edge_ids", ()))
        unknown_selected = sorted(set(selected_edges) - set(row_candidates))
        if unknown_selected:
            raise TensorizerViolation(f"selected edges not in candidate view: {unknown_selected}")
        time_steps = [
            int(actor_row.get("time_step", 0))
            for actor_row in row.get("actor_safe_rows", ())
            if isinstance(actor_row, Mapping)
        ]
        candidate_count = len(row_candidates)
        selected_count = len(selected_edges)
        feature_rows.append(
            (
                float(len(tuple(view.get("node_ids", ())))),
                float(candidate_count),
                float(selected_count),
                float(selected_count / candidate_count) if candidate_count else 0.0,
                float(sum(time_steps) / len(time_steps)) if time_steps else 0.0,
            )
        )
        edge_mask_rows.append(tuple(edge_id in set(row_candidates) for edge_id in candidate_ids))
    missing_candidates = [
        sorted(candidate_set - set(_critic_view(row).get("candidate_edge_ids", ())))
        for row in row_tuple
    ]
    if any(missing_candidates):
        # Variable edge sets are represented by masks, so this is not an error.
        pass
    return CriticTensorBatch(
        global_features=torch.tensor(feature_rows, dtype=torch.float32),
        edge_mask=torch.tensor(edge_mask_rows, dtype=torch.bool),
        candidate_edge_ids=candidate_ids,
    )


def _critic_view(row: Mapping[str, object]) -> Mapping[str, object]:
    view = row.get("critic_view")
    if not isinstance(view, Mapping):
        raise TensorizerViolation("evidence row must include critic_view mapping")
    if view.get("view_role") != "critic_centralized_training_only":
        raise TensorizerViolation("critic_view must be training-only")
    return view


def _neighbor_value(neighbor: object, name: str) -> object:
    if isinstance(neighbor, Mapping):
        if name not in neighbor:
            raise TensorizerViolation(f"missing neighbor field: {name}")
        return neighbor[name]
    if hasattr(neighbor, name):
        return getattr(neighbor, name)
    raise TensorizerViolation(f"missing neighbor field: {name}")


def _kind_flag(kind: str, expected: str) -> float:
    return 1.0 if kind.lower() == expected else 0.0


def _finite(name: str, value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TensorizerViolation(f"{name} must be finite") from exc
    if not isfinite(result):
        raise TensorizerViolation(f"{name} must be finite")
    return result
