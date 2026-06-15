"""Supervised dry-run batching from Stage 7 learning evidence."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from marl_topology.models.tensorizers import (
    ActorEdgeTensorBatch,
    CriticTensorBatch,
    TensorizerViolation,
    tensorize_actor_policy_inputs,
    tensorize_critic_evidence_rows,
)
from marl_topology.policies import ActorPolicyInput


STAGE10_SUPERVISED_BATCH_SCHEMA_ID = "stage10_supervised_batch_v1"
EDGE_DELTA_TARGET_FIELDS = (
    "delta_consensus_success_probability",
    "delta_latency",
    "delta_energy",
    "delta_feasibility",
)


class SupervisedBatchingViolation(ValueError):
    """Raised when supervised targets are missing or leak into actor inputs."""


@dataclass(frozen=True, slots=True)
class CriticTargetTensors:
    value: torch.Tensor
    feasibility: torch.Tensor
    consensus_success_probability: torch.Tensor
    latency: torch.Tensor
    energy: torch.Tensor
    edge_delta_add: torch.Tensor
    edge_delta_remove: torch.Tensor
    edge_delta_keep: torch.Tensor
    edge_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class SupervisedLearningBatch:
    actor_batch: ActorEdgeTensorBatch
    actor_edge_targets: torch.Tensor
    actor_edge_delta_targets: torch.Tensor
    critic_batch: CriticTensorBatch
    critic_targets: CriticTargetTensors
    dataset_id: str
    schema_id: str = STAGE10_SUPERVISED_BATCH_SCHEMA_ID

    def __post_init__(self) -> None:
        if self.schema_id != STAGE10_SUPERVISED_BATCH_SCHEMA_ID:
            raise SupervisedBatchingViolation("unexpected supervised batch schema")
        if self.actor_edge_targets.shape != (self.actor_batch.edge_count,):
            raise SupervisedBatchingViolation("actor target shape mismatch")
        if self.actor_edge_delta_targets.shape != (self.actor_batch.edge_count,):
            raise SupervisedBatchingViolation("actor edge-delta target shape mismatch")
        if self.critic_targets.edge_mask.shape != self.critic_batch.edge_mask.shape:
            raise SupervisedBatchingViolation("critic target mask shape mismatch")
        _assert_finite_tensor("actor_edge_targets", self.actor_edge_targets)
        _assert_finite_tensor("actor_edge_delta_targets", self.actor_edge_delta_targets)

    def target_summary(self) -> dict[str, object]:
        positives = int((self.actor_edge_targets > 0.5).sum().item())
        return {
            "schema_id": self.schema_id,
            "dataset_id": self.dataset_id,
            "actor_edge_count": self.actor_batch.edge_count,
            "actor_positive_edges": positives,
            "actor_negative_edges": self.actor_batch.edge_count - positives,
            "critic_batch_size": self.critic_batch.batch_size,
            "critic_edge_count": self.critic_batch.edge_count,
            "actor_safe_fields_separated_from_targets": True,
            "targets_feed_actor_model": False,
        }


def load_learning_evidence_json(path: str | Path) -> Mapping[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_supervised_batch_from_evidence(
    evidence: Mapping[str, object],
    *,
    row_limit: int | None = None,
) -> SupervisedLearningBatch:
    rows_raw = evidence.get("rows")
    if not isinstance(rows_raw, list) or not rows_raw:
        raise SupervisedBatchingViolation("evidence must contain nonempty rows")
    rows = tuple(rows_raw[:row_limit] if row_limit is not None else rows_raw)
    dataset_id = str(evidence.get("dataset_id", "unknown_learning_evidence"))
    return build_supervised_batch_from_evidence_rows(rows, dataset_id=dataset_id)


def build_supervised_batch_from_evidence_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    dataset_id: str,
) -> SupervisedLearningBatch:
    row_tuple = tuple(rows)
    if not row_tuple:
        raise SupervisedBatchingViolation("at least one evidence row is required")
    policy_inputs: list[ActorPolicyInput] = []
    actor_targets: list[float] = []
    actor_delta_targets: list[float] = []
    for row in row_tuple:
        selected_edges = _required_str_set(row, "selected_edges")
        delta_by_edge = _delta_consensus_by_edge(row)
        actor_rows = row.get("actor_safe_rows")
        if not isinstance(actor_rows, list) or not actor_rows:
            raise SupervisedBatchingViolation("row must include actor_safe_rows")
        for actor_row in actor_rows:
            if not isinstance(actor_row, Mapping):
                raise SupervisedBatchingViolation("actor_safe_rows must be mappings")
            policy_input = ActorPolicyInput.from_actor_safe_row(actor_row)
            for neighbor in policy_input.local_neighbor_observations:
                edge_id = str(_neighbor_value(neighbor, "edge_id"))
                actor_targets.append(1.0 if edge_id in selected_edges else 0.0)
                actor_delta_targets.append(float(delta_by_edge.get(edge_id, 0.0)))
            policy_inputs.append(policy_input)
    actor_batch = tensorize_actor_policy_inputs(policy_inputs)
    if len(actor_targets) != actor_batch.edge_count:
        raise SupervisedBatchingViolation("actor target count does not match edge tensors")
    critic_batch = tensorize_critic_evidence_rows(row_tuple)
    critic_targets = _build_critic_targets(row_tuple, critic_batch)
    return SupervisedLearningBatch(
        actor_batch=actor_batch,
        actor_edge_targets=torch.tensor(actor_targets, dtype=torch.float32),
        actor_edge_delta_targets=torch.tensor(actor_delta_targets, dtype=torch.float32),
        critic_batch=critic_batch,
        critic_targets=critic_targets,
        dataset_id=dataset_id,
    )


def _build_critic_targets(
    rows: tuple[Mapping[str, object], ...],
    critic_batch: CriticTensorBatch,
) -> CriticTargetTensors:
    value: list[float] = []
    feasibility: list[float] = []
    consensus: list[float] = []
    latency: list[float] = []
    energy: list[float] = []
    add_rows: list[list[float]] = []
    remove_rows: list[list[float]] = []
    keep_rows: list[list[float]] = []
    for row in rows:
        consensus_value = _required_float(row, "consensus_success_probability")
        latency_value = _required_float(row, "latency")
        energy_value = _required_float(row, "energy")
        value.append(consensus_value)
        feasibility.append(1.0 if bool(row.get("feasible_under_tau_requirement")) else 0.0)
        consensus.append(consensus_value)
        latency.append(latency_value)
        energy.append(energy_value)
        targets = _edge_delta_maps(row)
        add_rows.append([targets["add_edge"].get(edge_id, 0.0) for edge_id in critic_batch.candidate_edge_ids])
        remove_rows.append(
            [targets["remove_edge"].get(edge_id, 0.0) for edge_id in critic_batch.candidate_edge_ids]
        )
        keep_rows.append(
            [targets["keep_edge"].get(edge_id, 0.0) for edge_id in critic_batch.candidate_edge_ids]
        )
    return CriticTargetTensors(
        value=torch.tensor(value, dtype=torch.float32),
        feasibility=torch.tensor(feasibility, dtype=torch.float32),
        consensus_success_probability=torch.tensor(consensus, dtype=torch.float32),
        latency=torch.tensor(latency, dtype=torch.float32),
        energy=torch.tensor(energy, dtype=torch.float32),
        edge_delta_add=torch.tensor(add_rows, dtype=torch.float32),
        edge_delta_remove=torch.tensor(remove_rows, dtype=torch.float32),
        edge_delta_keep=torch.tensor(keep_rows, dtype=torch.float32),
        edge_mask=critic_batch.edge_mask.clone(),
    )


def _edge_delta_maps(row: Mapping[str, object]) -> dict[str, dict[str, float]]:
    learning_targets = row.get("learning_targets")
    if not isinstance(learning_targets, list) or not learning_targets:
        raise SupervisedBatchingViolation("row must include learning_targets")
    maps: dict[str, dict[str, float]] = {
        "add_edge": {},
        "remove_edge": {},
        "keep_edge": {},
    }
    for target in learning_targets:
        if not isinstance(target, Mapping):
            raise SupervisedBatchingViolation("learning targets must be mappings")
        action_type = str(target.get("action_type"))
        if action_type not in maps:
            continue
        edge_id = str(target.get("edge_id", ""))
        if not edge_id:
            raise SupervisedBatchingViolation("learning target missing edge_id")
        maps[action_type][edge_id] = _required_float(
            target,
            "delta_consensus_success_probability",
        )
    return maps


def _delta_consensus_by_edge(row: Mapping[str, object]) -> dict[str, float]:
    targets = _edge_delta_maps(row)
    selected_edges = _required_str_set(row, "selected_edges")
    result: dict[str, float] = {}
    for edge_id, delta in targets["add_edge"].items():
        if edge_id not in selected_edges:
            result[edge_id] = delta
    for edge_id, delta in targets["remove_edge"].items():
        if edge_id in selected_edges:
            result[edge_id] = -delta
    for edge_id, delta in targets["keep_edge"].items():
        result.setdefault(edge_id, delta)
    return result


def _required_str_set(row: Mapping[str, object], field: str) -> set[str]:
    value = row.get(field)
    if not isinstance(value, list):
        raise SupervisedBatchingViolation(f"row missing target field: {field}")
    return {str(item) for item in value}


def _required_float(row: Mapping[str, object], field: str) -> float:
    if field not in row:
        raise SupervisedBatchingViolation(f"row missing target field: {field}")
    try:
        return float(row[field])
    except (TypeError, ValueError) as exc:
        raise SupervisedBatchingViolation(f"target field must be numeric: {field}") from exc


def _neighbor_value(neighbor: object, name: str) -> object:
    if isinstance(neighbor, Mapping):
        if name not in neighbor:
            raise SupervisedBatchingViolation(f"missing neighbor field: {name}")
        return neighbor[name]
    if hasattr(neighbor, name):
        return getattr(neighbor, name)
    raise SupervisedBatchingViolation(f"missing neighbor field: {name}")


def _assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all().item():
        raise TensorizerViolation(f"{name} contains non-finite values")
