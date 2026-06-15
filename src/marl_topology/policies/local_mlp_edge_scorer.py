"""Stage 9.0 local MLP edge scorer scaffold.

The scorer maps actor-safe local observations to directed edge scores. It does
not provide fitting, learner updates, checkpoint IO, or artifact writes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import marl_topology.training.torch_backend as _torch_backend
from .actor_interface import ActorPolicyInput, ActorPolicyOutput
from .edge_scores import EdgeScoreBatch, EdgeScoreRecord, make_directed_edge_id

torch = _torch_backend.torch
nn = torch.nn


STAGE9_0_LOCAL_MLP_STAGE_ID = (
    "stage_9_0_local_mlp_edge_scorer_baseline_without_training_execution"
)
LOCAL_MLP_EDGE_SCORER_ID = "local_mlp_edge_scorer_v1_untrained_scaffold"
LOCAL_MLP_FEATURE_SCHEMA_ID = "local_mlp_edge_feature_v1"
LOCAL_MLP_SCORE_SOURCE = "stage9_0_local_mlp_edge_scorer_untrained"
LOCAL_MLP_EDGE_FEATURE_FIELDS = (
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


class LocalMLPEdgeScorerViolation(ValueError):
    """Raised when Stage 9.0 local MLP scaffold boundaries are violated."""


@dataclass(frozen=True, slots=True)
class LocalMLPEdgeScorerConfig:
    scorer_id: str = LOCAL_MLP_EDGE_SCORER_ID
    feature_schema_id: str = LOCAL_MLP_FEATURE_SCHEMA_ID
    input_dim: int = len(LOCAL_MLP_EDGE_FEATURE_FIELDS)
    hidden_dim: int = 16
    output_dim: int = 1
    include_probability: bool = True
    freeze_parameters_for_stage9: bool = True
    training_execution_allowed: bool = False
    checkpoint_io_allowed: bool = False
    artifact_writes_allowed: bool = False

    def __post_init__(self) -> None:
        if self.feature_schema_id != LOCAL_MLP_FEATURE_SCHEMA_ID:
            raise LocalMLPEdgeScorerViolation("unexpected feature schema id")
        if self.input_dim != len(LOCAL_MLP_EDGE_FEATURE_FIELDS):
            raise LocalMLPEdgeScorerViolation("input_dim must match local feature schema")
        if self.hidden_dim <= 0:
            raise LocalMLPEdgeScorerViolation("hidden_dim must be positive")
        if self.output_dim != 1:
            raise LocalMLPEdgeScorerViolation("output_dim must be 1")
        if self.training_execution_allowed:
            raise LocalMLPEdgeScorerViolation("Stage 9.0 does not allow training execution")
        if self.checkpoint_io_allowed:
            raise LocalMLPEdgeScorerViolation("Stage 9.0 does not allow checkpoint IO")
        if self.artifact_writes_allowed:
            raise LocalMLPEdgeScorerViolation("Stage 9.0 does not allow artifact writes")


@dataclass(frozen=True, slots=True)
class EncodedLocalEdge:
    neighbor_id: str
    edge_id: str
    directed_edge_id: str
    features: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.features) != len(LOCAL_MLP_EDGE_FEATURE_FIELDS):
            raise LocalMLPEdgeScorerViolation("encoded edge feature dimension mismatch")


class LocalMLPEdgeScorer(nn.Module):
    """Minimal local MLP edge-score scaffold for actor-safe observations."""

    def __init__(self, config: LocalMLPEdgeScorerConfig | None = None) -> None:
        super().__init__()
        self.config = config or LocalMLPEdgeScorerConfig()
        self.network = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.output_dim),
        )
        if self.config.freeze_parameters_for_stage9:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        if edge_features.ndim != 2 or edge_features.shape[1] != self.config.input_dim:
            raise LocalMLPEdgeScorerViolation("edge_features shape must be [N, input_dim]")
        return self.network(edge_features).reshape(-1)

    def score_actor_input(self, policy_input: ActorPolicyInput) -> ActorPolicyOutput:
        encoded_edges = encode_actor_local_edges(policy_input)
        if not encoded_edges:
            batch = EdgeScoreBatch.from_records(
                (),
                batch_id=f"{self.config.scorer_id}:{policy_input.agent_id}:{policy_input.time_step}",
                source=LOCAL_MLP_SCORE_SOURCE,
            )
            return ActorPolicyOutput(agent_id=policy_input.agent_id, edge_scores=batch)

        feature_tensor = torch.tensor(
            [edge.features for edge in encoded_edges],
            dtype=torch.float32,
        )
        with torch.no_grad():
            logits = self.forward(feature_tensor).detach().cpu().tolist()
        probabilities = (
            torch.sigmoid(torch.tensor(logits, dtype=torch.float32)).tolist()
            if self.config.include_probability
            else [None for _ in logits]
        )
        records = tuple(
            EdgeScoreRecord(
                agent_id=policy_input.agent_id,
                neighbor_id=edge.neighbor_id,
                edge_id=edge.edge_id,
                directed_edge_id=edge.directed_edge_id,
                score=float(logit),
                probability=(
                    float(probabilities[index])
                    if probabilities[index] is not None
                    else None
                ),
                score_source=LOCAL_MLP_SCORE_SOURCE,
                time_step=policy_input.time_step,
            )
            for index, (edge, logit) in enumerate(zip(encoded_edges, logits, strict=True))
        )
        batch = EdgeScoreBatch.from_records(
            records,
            batch_id=f"{self.config.scorer_id}:{policy_input.agent_id}:{policy_input.time_step}",
            source=LOCAL_MLP_SCORE_SOURCE,
        )
        return ActorPolicyOutput(agent_id=policy_input.agent_id, edge_scores=batch)

    def training_boundary(self) -> dict[str, object]:
        return {
            "stage": STAGE9_0_LOCAL_MLP_STAGE_ID,
            "scorer_id": self.config.scorer_id,
            "feature_schema_id": self.config.feature_schema_id,
            "training_execution_allowed": False,
            "checkpoint_io_allowed": False,
            "artifact_writes_allowed": False,
            "parameters_frozen_for_stage9": self.config.freeze_parameters_for_stage9,
            "outputs_edge_scores_only": True,
        }


def encode_actor_local_edges(policy_input: ActorPolicyInput) -> tuple[EncodedLocalEdge, ...]:
    records: list[EncodedLocalEdge] = []
    for neighbor in policy_input.local_neighbor_observations:
        neighbor_id = str(_neighbor_value(neighbor, "neighbor_id"))
        edge_id = str(_neighbor_value(neighbor, "edge_id"))
        directed_edge_id = make_directed_edge_id(policy_input.agent_id, neighbor_id)
        features = (
            _kind_flag(policy_input.agent_kind, "vehicle"),
            _kind_flag(policy_input.agent_kind, "rsu"),
            _kind_flag(str(_neighbor_value(neighbor, "neighbor_kind")), "vehicle"),
            _kind_flag(str(_neighbor_value(neighbor, "neighbor_kind")), "rsu"),
            float(policy_input.local_position_m[0]),
            float(policy_input.local_position_m[1]),
            float(policy_input.local_position_m[2]),
            float(_neighbor_value(neighbor, "distance_3d_m")),
            float(_neighbor_value(neighbor, "link_success_probability")),
            float(_neighbor_value(neighbor, "estimated_link_latency_s")),
            float(_neighbor_value(neighbor, "estimated_link_energy_j")),
        )
        records.append(
            EncodedLocalEdge(
                neighbor_id=neighbor_id,
                edge_id=edge_id,
                directed_edge_id=directed_edge_id,
                features=features,
            )
        )
    return tuple(sorted(records, key=lambda record: record.directed_edge_id))


def build_stage9_0_local_mlp_report() -> dict[str, object]:
    return {
        "stage": STAGE9_0_LOCAL_MLP_STAGE_ID,
        "scorer_id": LOCAL_MLP_EDGE_SCORER_ID,
        "feature_schema_id": LOCAL_MLP_FEATURE_SCHEMA_ID,
        "feature_fields": list(LOCAL_MLP_EDGE_FEATURE_FIELDS),
        "active_actor_output_schema": "actor_policy_local_edge_score_output_v1",
        "legacy_activate_schema": "actor_policy_local_edge_decision_v1",
        "outputs_edge_scores_only": True,
        "activate_output_allowed": False,
        "training_execution_allowed": False,
        "checkpoint_io_allowed": False,
        "artifact_writes_allowed": False,
        "ppo_mappo_coma_implemented": False,
        "graph_recurrent_attention_modules_implemented": False,
    }


def _neighbor_value(neighbor: object, name: str) -> object:
    if isinstance(neighbor, Mapping):
        if name not in neighbor:
            raise LocalMLPEdgeScorerViolation(f"missing neighbor field: {name}")
        return neighbor[name]
    if hasattr(neighbor, name):
        return getattr(neighbor, name)
    raise LocalMLPEdgeScorerViolation(f"missing neighbor field: {name}")


def _kind_flag(kind: str, expected: str) -> float:
    return 1.0 if kind.lower() == expected else 0.0
