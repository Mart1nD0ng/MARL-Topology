"""Training-only critic feature schema for Stage 27 value repair."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

import torch

from marl_topology.data.stage21_objective_stack_evidence import Stage21EvaluationContext
from marl_topology.policies.interface_contract import ACTOR_POLICY_FORBIDDEN_INPUT_FIELDS


STAGE27_VALUE_CRITIC_FEATURE_SCHEMA_ID = "stage27_pre_action_value_critic_features_v1"
STAGE27_DIAGNOSTIC_CRITIC_FEATURE_SCHEMA_ID = "stage27_action_conditioned_diagnostic_features_v1"

VALUE_REJECTION_REASON_FIELDS = (
    "previous_rejection_tx_budget_exceeded",
    "previous_rejection_rx_capacity_exceeded",
    "previous_rejection_interference_conflict",
    "previous_rejection_role_not_allowed",
    "previous_rejection_duplicate_edge",
    "previous_rejection_other",
)

VALUE_CRITIC_FEATURE_FIELDS = (
    "node_count",
    "vehicle_count",
    "rsu_count",
    "base_station_count",
    "candidate_edge_count",
    "previous_selected_edge_count",
    "previous_selected_edge_density",
    "previous_component_count",
    "previous_largest_component_size",
    "previous_vehicle_vehicle_edge_count",
    "previous_vehicle_rsu_edge_count",
    "previous_base_station_edge_count",
    "candidate_link_success_mean",
    "candidate_link_success_min",
    "candidate_link_success_std",
    "candidate_deadline_delivery_mean",
    "candidate_deadline_delivery_min",
    "candidate_latency_mean",
    "candidate_latency_max",
    "candidate_energy_mean",
    "candidate_energy_sum",
    "required_reliability_met_rate",
    "required_transmission_time_capped_rate",
    "expected_attempts_mean",
    "expected_attempts_max",
    "tx_budget_total",
    "tx_budget_used_previous",
    "tx_budget_remaining_estimate",
    "rx_capacity_total",
    "rx_capacity_used_previous",
    "conflict_group_count",
    "channel_slot_count",
    "previous_projection_rejection_rate",
    "previous_top_rejection_rate",
    *VALUE_REJECTION_REASON_FIELDS,
    "time_step",
    "scenario_family_code",
    "mobility_step_index",
    "seed_group_code",
    "previous_reward_summary",
    "previous_consensus_success_probability",
    "previous_latency",
    "previous_energy",
    "previous_violation_indicator",
)

VALUE_FORBIDDEN_CURRENT_FIELDS = (
    "current_consensus_success_probability",
    "current_latency",
    "current_energy",
    "current_reward_surrogate",
    "current_selected_edge_count",
    "current_selected_edge_density",
)

DIAGNOSTIC_FEATURE_FIELDS = (
    "current_consensus_success_probability",
    "current_latency",
    "current_energy",
    "current_reward_surrogate",
    "current_selected_edge_count",
    "current_selected_edge_density",
    "current_top_proposal_rejection_rate",
    "current_above_threshold_rejection_rate",
    "current_violation_indicator",
    "edge_delta_target_mean",
    "edge_delta_target_std",
)


class CriticFeatureViolation(ValueError):
    """Raised when critic features cross value/diagnostic boundaries."""


@dataclass(frozen=True, slots=True)
class PreviousStepSummary:
    selected_edges: tuple[str, ...] = ()
    projection_rejection_rate: float = 0.0
    top_rejection_rate: float = 0.0
    rejection_by_reason: Mapping[str, int] | None = None
    reward_summary: float = 0.0
    consensus_success_probability: float = 0.0
    latency: float = 0.0
    energy: float = 0.0
    violation_indicator: float = 1.0

    def reasons(self) -> Mapping[str, int]:
        return self.rejection_by_reason or {}


@dataclass(frozen=True, slots=True)
class CriticValueFeatureRecord:
    values: tuple[float, ...]
    metadata: Mapping[str, object]
    feature_schema_id: str = STAGE27_VALUE_CRITIC_FEATURE_SCHEMA_ID
    role: str = "mappo_value_pre_action"
    action_independent: bool = True
    actor_input_allowed: bool = False

    def __post_init__(self) -> None:
        if self.feature_schema_id != STAGE27_VALUE_CRITIC_FEATURE_SCHEMA_ID:
            raise CriticFeatureViolation("unexpected value feature schema")
        if len(self.values) != len(VALUE_CRITIC_FEATURE_FIELDS):
            raise CriticFeatureViolation("value feature dimensionality mismatch")
        if any(not isfinite(value) for value in self.values):
            raise CriticFeatureViolation("value features must be finite")
        if not self.action_independent:
            raise CriticFeatureViolation("value critic record must be action independent")
        if self.actor_input_allowed:
            raise CriticFeatureViolation("value critic features are critic-only")
        forbidden = sorted(set(self.metadata) & set(VALUE_FORBIDDEN_CURRENT_FIELDS))
        if forbidden:
            raise CriticFeatureViolation(f"value critic metadata has post-action fields: {forbidden}")

    def as_mapping(self) -> dict[str, float]:
        return dict(zip(VALUE_CRITIC_FEATURE_FIELDS, self.values, strict=True))


@dataclass(frozen=True, slots=True)
class CriticDiagnosticFeatureRecord:
    values: tuple[float, ...]
    metadata: Mapping[str, object]
    feature_schema_id: str = STAGE27_DIAGNOSTIC_CRITIC_FEATURE_SCHEMA_ID
    role: str = "action_conditioned_diagnostic"
    action_independent: bool = False
    actor_input_allowed: bool = False

    def __post_init__(self) -> None:
        if self.feature_schema_id != STAGE27_DIAGNOSTIC_CRITIC_FEATURE_SCHEMA_ID:
            raise CriticFeatureViolation("unexpected diagnostic feature schema")
        if len(self.values) != len(DIAGNOSTIC_FEATURE_FIELDS):
            raise CriticFeatureViolation("diagnostic feature dimensionality mismatch")
        if any(not isfinite(value) for value in self.values):
            raise CriticFeatureViolation("diagnostic features must be finite")
        if self.action_independent:
            raise CriticFeatureViolation("diagnostic critic record must be action conditioned")
        if self.actor_input_allowed:
            raise CriticFeatureViolation("diagnostic features are critic-only")

    def as_mapping(self) -> dict[str, float]:
        return dict(zip(DIAGNOSTIC_FEATURE_FIELDS, self.values, strict=True))


@dataclass(frozen=True, slots=True)
class CriticFeatureBatch:
    value_features: torch.Tensor
    diagnostic_features: torch.Tensor
    value_records: tuple[CriticValueFeatureRecord, ...]
    diagnostic_records: tuple[CriticDiagnosticFeatureRecord, ...]

    def __post_init__(self) -> None:
        if self.value_features.ndim != 2:
            raise CriticFeatureViolation("value feature tensor must be rank 2")
        if self.value_features.shape[1] != len(VALUE_CRITIC_FEATURE_FIELDS):
            raise CriticFeatureViolation("value feature tensor dimension mismatch")
        if self.diagnostic_features.ndim != 2:
            raise CriticFeatureViolation("diagnostic feature tensor must be rank 2")
        if self.diagnostic_features.shape[1] != len(DIAGNOSTIC_FEATURE_FIELDS):
            raise CriticFeatureViolation("diagnostic feature tensor dimension mismatch")
        if self.value_features.shape[0] != len(self.value_records):
            raise CriticFeatureViolation("value record count mismatch")
        if self.diagnostic_features.shape[0] != len(self.diagnostic_records):
            raise CriticFeatureViolation("diagnostic record count mismatch")
        if not torch.isfinite(self.value_features).all().item():
            raise CriticFeatureViolation("value feature tensor must be finite")
        if not torch.isfinite(self.diagnostic_features).all().item():
            raise CriticFeatureViolation("diagnostic feature tensor must be finite")

    @property
    def batch_size(self) -> int:
        return int(self.value_features.shape[0])


def build_value_feature_record(
    *,
    row: object,
    context: Stage21EvaluationContext,
    previous: PreviousStepSummary,
    step_index: int,
    seed: int,
) -> CriticValueFeatureRecord:
    actor_rows = tuple(row.actor_safe_view)
    _reject_actor_forbidden_fields(actor_rows)
    candidate_edges = tuple(context.graph.edge_ids)
    selected_edges = tuple(previous.selected_edges)
    node_ids = tuple(context.graph.node_ids)
    role_counts = _role_counts(node_ids)
    link_success = _link_values(context, candidate_edges, "link_success_probability")
    latency = _link_values(context, candidate_edges, "latency_s")
    energy = _link_values(context, candidate_edges, "energy_j")
    deadline_delivery = _actor_values(actor_rows, "estimated_deadline_delivery_probability")
    expected_attempts = _actor_values(actor_rows, "estimated_retransmission_attempts")
    required_met = [1.0 if value >= 0.9 else 0.0 for value in link_success]
    component_count, largest_component = _component_stats(node_ids, selected_edges)
    role_edges = _selected_role_edge_counts(selected_edges)
    tx_budget_total = _sum_actor_field(actor_rows, "tx_budget_used") + _sum_actor_field(
        actor_rows, "tx_budget_remaining"
    )
    rx_capacity_total = _sum_actor_field(actor_rows, "rx_capacity_estimate_for_neighbor")
    rejection_values = _reason_feature_values(previous.reasons())
    candidate_count = len(candidate_edges)
    selected_count = len(selected_edges)
    values = (
        float(len(node_ids)),
        float(role_counts["vehicle"]),
        float(role_counts["rsu"]),
        float(role_counts["base_station"]),
        float(candidate_count),
        float(selected_count),
        float(selected_count / candidate_count) if candidate_count else 0.0,
        float(component_count),
        float(largest_component),
        float(role_edges["vehicle_vehicle"]),
        float(role_edges["vehicle_rsu"]),
        float(role_edges["base_station"]),
        _mean(link_success),
        min(link_success) if link_success else 0.0,
        _std(link_success),
        _mean(deadline_delivery),
        min(deadline_delivery) if deadline_delivery else 0.0,
        _mean(latency),
        max(latency) if latency else 0.0,
        _mean(energy),
        sum(energy),
        _mean(required_met),
        0.0,
        _mean(expected_attempts),
        max(expected_attempts) if expected_attempts else 0.0,
        tx_budget_total,
        _sum_actor_field(actor_rows, "tx_budget_used"),
        _sum_actor_field(actor_rows, "tx_budget_remaining"),
        rx_capacity_total,
        _sum_actor_field(actor_rows, "local_incoming_degree_estimate"),
        float(len({str(item.get("local_conflict_group_occupancy", 0)) for item in actor_rows})),
        float(sum(1 for item in actor_rows if bool(item.get("channel_slot_available", False)))),
        float(previous.projection_rejection_rate),
        float(previous.top_rejection_rate),
        *rejection_values,
        float(step_index),
        _stable_code(str(context.fixture.fixture_id)),
        float(context.time_step),
        _stable_code(str(seed // 1000)),
        float(previous.reward_summary),
        float(previous.consensus_success_probability),
        float(previous.latency),
        float(previous.energy),
        float(previous.violation_indicator),
    )
    return CriticValueFeatureRecord(
        values=values,
        metadata={
            "scenario_id": str(context.fixture.fixture_id),
            "time_step": int(step_index),
            "seed": int(seed),
            "candidate_edge_ids": candidate_edges,
            "previous_selected_edges": selected_edges,
        },
    )


def build_diagnostic_feature_record(
    *,
    row: object,
    context: Stage21EvaluationContext,
    selected_edges: Sequence[str],
    consensus_success_probability: float,
    latency: float,
    energy: float,
    reward_surrogate: float,
    projection_diagnostics: Mapping[str, object],
) -> CriticDiagnosticFeatureRecord:
    candidate_count = len(tuple(context.graph.edge_ids))
    selected_count = len(tuple(selected_edges))
    edge_deltas = [
        float(item.get("delta_consensus_success_probability", 0.0))
        for item in row.critic_target_view.get("edge_delta_targets", ())
        if isinstance(item, Mapping)
    ]
    values = (
        float(consensus_success_probability),
        float(latency),
        float(energy),
        float(reward_surrogate),
        float(selected_count),
        float(selected_count / candidate_count) if candidate_count else 0.0,
        float(projection_diagnostics.get("top_proposal_rejection_rate", 0.0)),
        float(projection_diagnostics.get("above_threshold_rejection_rate", 0.0)),
        1.0 if float(consensus_success_probability) < 0.9 else 0.0,
        _mean(edge_deltas),
        _std(edge_deltas),
    )
    return CriticDiagnosticFeatureRecord(
        values=values,
        metadata={
            "current_consensus_success_probability": float(consensus_success_probability),
            "current_latency": float(latency),
            "current_energy": float(energy),
            "current_reward_surrogate": float(reward_surrogate),
            "selected_edges": tuple(str(edge_id) for edge_id in selected_edges),
        },
    )


def build_feature_batch(
    value_records: Iterable[CriticValueFeatureRecord],
    diagnostic_records: Iterable[CriticDiagnosticFeatureRecord],
) -> CriticFeatureBatch:
    value_tuple = tuple(value_records)
    diagnostic_tuple = tuple(diagnostic_records)
    if len(value_tuple) != len(diagnostic_tuple):
        raise CriticFeatureViolation("value and diagnostic record counts must match")
    return CriticFeatureBatch(
        value_features=torch.tensor([record.values for record in value_tuple], dtype=torch.float32),
        diagnostic_features=torch.tensor(
            [record.values for record in diagnostic_tuple],
            dtype=torch.float32,
        ),
        value_records=value_tuple,
        diagnostic_records=diagnostic_tuple,
    )


def value_feature_health(records: Iterable[CriticValueFeatureRecord]) -> dict[str, object]:
    record_tuple = tuple(records)
    if not record_tuple:
        return {"record_count": 0, "missingness": 1.0, "feature_dim": len(VALUE_CRITIC_FEATURE_FIELDS)}
    values = [value for record in record_tuple for value in record.values]
    zero_count = sum(1 for value in values if value == 0.0)
    return {
        "record_count": len(record_tuple),
        "feature_dim": len(VALUE_CRITIC_FEATURE_FIELDS),
        "all_finite": all(isfinite(value) for value in values),
        "zero_fraction": zero_count / len(values) if values else 0.0,
        "forbidden_current_fields": list(VALUE_FORBIDDEN_CURRENT_FIELDS),
        "actor_input_allowed": False,
    }


def _reject_actor_forbidden_fields(actor_rows: Sequence[Mapping[str, object]]) -> None:
    forbidden = sorted(
        {
            field
            for actor_row in actor_rows
            for field in actor_row
            if field in ACTOR_POLICY_FORBIDDEN_INPUT_FIELDS
        }
    )
    if forbidden:
        raise CriticFeatureViolation(f"actor rows contain forbidden fields: {forbidden}")


def _link_values(context: Stage21EvaluationContext, edge_ids: Sequence[str], field: str) -> list[float]:
    values = []
    for edge_id in edge_ids:
        record = context.link_records.get(edge_id)
        if record is None:
            values.append(0.0)
        else:
            values.append(float(getattr(record, field)))
    return values


def _actor_values(actor_rows: Sequence[Mapping[str, object]], field: str) -> list[float]:
    return [float(row.get(field, 0.0)) for row in actor_rows]


def _sum_actor_field(actor_rows: Sequence[Mapping[str, object]], field: str) -> float:
    return sum(float(row.get(field, 0.0)) for row in actor_rows)


def _role_counts(node_ids: Sequence[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for node_id in node_ids:
        counts[_node_role(node_id)] += 1
    return counts


def _selected_role_edge_counts(selected_edges: Sequence[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for edge_id in selected_edges:
        left, right = _split_edge(edge_id)
        roles = sorted((_node_role(left), _node_role(right)))
        if roles == ["vehicle", "vehicle"]:
            counts["vehicle_vehicle"] += 1
        elif "base_station" in roles:
            counts["base_station"] += 1
        elif "rsu" in roles and "vehicle" in roles:
            counts["vehicle_rsu"] += 1
    return counts


def _component_stats(node_ids: Sequence[str], selected_edges: Sequence[str]) -> tuple[int, int]:
    parent = {node_id: node_id for node_id in node_ids}

    def find(node_id: str) -> str:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    node_set = set(node_ids)
    for edge_id in selected_edges:
        left, right = _split_edge(edge_id)
        if left in node_set and right in node_set:
            union(left, right)
    groups: defaultdict[str, int] = defaultdict(int)
    for node_id in node_ids:
        groups[find(node_id)] += 1
    if not groups:
        return 0, 0
    return len(groups), max(groups.values())


def _split_edge(edge_id: str) -> tuple[str, str]:
    parts = str(edge_id).split("--")
    if len(parts) != 2:
        return str(edge_id), ""
    return parts[0], parts[1]


def _node_role(node_id: str) -> str:
    lowered = str(node_id).lower()
    if lowered.startswith("veh"):
        return "vehicle"
    if lowered.startswith("rsu"):
        return "rsu"
    if lowered.startswith("bs") or "base" in lowered:
        return "base_station"
    return "unknown"


def _reason_feature_values(reasons: Mapping[str, int]) -> tuple[float, ...]:
    other = 0
    values = []
    mapping = {
        "tx_budget_exceeded": "previous_rejection_tx_budget_exceeded",
        "rx_capacity_exceeded": "previous_rejection_rx_capacity_exceeded",
        "interference_conflict": "previous_rejection_interference_conflict",
        "role_not_allowed": "previous_rejection_role_not_allowed",
        "duplicate_edge": "previous_rejection_duplicate_edge",
    }
    for reason_key in mapping:
        values.append(float(reasons.get(reason_key, 0)))
    known = set(mapping)
    for key, value in reasons.items():
        if key not in known:
            other += int(value)
    values.append(float(other))
    return tuple(values)


def _stable_code(value: str) -> float:
    return float(sum(ord(char) for char in value) % 997) / 997.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
