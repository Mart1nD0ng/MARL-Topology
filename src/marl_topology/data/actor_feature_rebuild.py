"""Stage 18 actor-safe local feature rebuild.

The rows built here are edge-level actor views. They extend the original
actor observation with local topology/projection/resource/message/link
summaries only. Global topology, global objectives, oracle outputs, future
outcomes, and critic-only targets are rejected.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite

from marl_topology.env import ACTOR_FORBIDDEN_FIELDS

from .actor_label_disambiguation import _neighbor_payload
from .learning_evidence import ACTOR_VIEW_FORBIDDEN_FIELDS, LearningEvidenceRow


STAGE18_FEATURE_SCHEMA_ID = "stage18_actor_safe_local_feature_schema_v1"

FEATURE_SOURCE_LOCAL_OBSERVATION = "local_observation"
FEATURE_SOURCE_LOCAL_HISTORY = "local_history"
FEATURE_SOURCE_LOCAL_PROJECTION_FEEDBACK = "local_projection_feedback"
FEATURE_SOURCE_LOCAL_MESSAGE = "local_message"
FEATURE_SOURCE_LOCAL_LINK_ESTIMATE = "local_link_estimate"

ALLOWED_FEATURE_SOURCES = frozenset(
    {
        FEATURE_SOURCE_LOCAL_OBSERVATION,
        FEATURE_SOURCE_LOCAL_HISTORY,
        FEATURE_SOURCE_LOCAL_PROJECTION_FEEDBACK,
        FEATURE_SOURCE_LOCAL_MESSAGE,
        FEATURE_SOURCE_LOCAL_LINK_ESTIMATE,
    }
)

STAGE18_FORBIDDEN_ACTOR_FIELDS = frozenset(
    {
        *ACTOR_FORBIDDEN_FIELDS,
        *ACTOR_VIEW_FORBIDDEN_FIELDS,
        "global_objective",
        "edge_delta_global_target",
        "future_outcome",
        "oracle_topology_membership",
    }
)

# Route B: per-kind tx/rx budgets come from the single source of truth in
# marl_topology.budgets so the actor's deployment assembler and the teacher's
# feasibility filter can never disagree. rsu=8 (was 4) lets an RSU host the N-1
# hub the only feasible consensus structure requires under interference.
from marl_topology.budgets import CANONICAL_ENDPOINT_BUDGET_BY_KIND as _CANONICAL_BUDGET

DEFAULT_TX_BUDGET_BY_KIND = dict(_CANONICAL_BUDGET)
DEFAULT_RX_CAPACITY_BY_KIND = dict(_CANONICAL_BUDGET)


class Stage18ActorFeatureViolation(ValueError):
    """Raised when rebuilt actor features cross the actor boundary."""


@dataclass(frozen=True, slots=True)
class Stage18FeatureSpec:
    """Actor-safe feature declaration used by docs and tests."""

    name: str
    group: str
    source: str
    actor_safe: bool = True

    def __post_init__(self) -> None:
        if self.source not in ALLOWED_FEATURE_SOURCES:
            raise Stage18ActorFeatureViolation(f"unknown feature source: {self.source}")
        if not self.actor_safe:
            raise Stage18ActorFeatureViolation(f"feature is not actor-safe: {self.name}")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "group": self.group,
            "source": self.source,
            "actor_safe": self.actor_safe,
        }


STAGE18_ACTOR_SAFE_FEATURE_SPECS = {
    spec.name: spec
    for spec in (
        Stage18FeatureSpec("edge_active_prev", "local_topology_history", FEATURE_SOURCE_LOCAL_HISTORY),
        Stage18FeatureSpec(
            "edge_active_current_local",
            "local_topology_history",
            FEATURE_SOURCE_LOCAL_PROJECTION_FEEDBACK,
        ),
        Stage18FeatureSpec(
            "local_outgoing_degree_prev",
            "local_topology_history",
            FEATURE_SOURCE_LOCAL_HISTORY,
        ),
        Stage18FeatureSpec(
            "local_incoming_degree_estimate",
            "local_topology_history",
            FEATURE_SOURCE_LOCAL_PROJECTION_FEEDBACK,
        ),
        Stage18FeatureSpec(
            "local_selected_edge_count",
            "local_topology_history",
            FEATURE_SOURCE_LOCAL_PROJECTION_FEEDBACK,
        ),
        Stage18FeatureSpec("last_action_proposed", "local_topology_history", FEATURE_SOURCE_LOCAL_HISTORY),
        Stage18FeatureSpec("last_action_accepted", "local_topology_history", FEATURE_SOURCE_LOCAL_HISTORY),
        Stage18FeatureSpec(
            "last_projection_rejection_reason",
            "local_topology_history",
            FEATURE_SOURCE_LOCAL_PROJECTION_FEEDBACK,
        ),
        Stage18FeatureSpec(
            "recent_rejection_reason_histogram",
            "local_topology_history",
            FEATURE_SOURCE_LOCAL_PROJECTION_FEEDBACK,
        ),
        Stage18FeatureSpec(
            "previous_selected_neighbor_summary",
            "local_topology_history",
            FEATURE_SOURCE_LOCAL_HISTORY,
        ),
        Stage18FeatureSpec("tx_budget_used", "local_resource_conflict_context", FEATURE_SOURCE_LOCAL_PROJECTION_FEEDBACK),
        Stage18FeatureSpec(
            "tx_budget_remaining",
            "local_resource_conflict_context",
            FEATURE_SOURCE_LOCAL_PROJECTION_FEEDBACK,
        ),
        Stage18FeatureSpec(
            "rx_capacity_estimate_for_neighbor",
            "local_resource_conflict_context",
            FEATURE_SOURCE_LOCAL_MESSAGE,
        ),
        Stage18FeatureSpec(
            "channel_slot_available",
            "local_resource_conflict_context",
            FEATURE_SOURCE_LOCAL_LINK_ESTIMATE,
        ),
        Stage18FeatureSpec(
            "local_channel_slot_occupancy",
            "local_resource_conflict_context",
            FEATURE_SOURCE_LOCAL_PROJECTION_FEEDBACK,
        ),
        Stage18FeatureSpec(
            "local_conflict_group_occupancy",
            "local_resource_conflict_context",
            FEATURE_SOURCE_LOCAL_PROJECTION_FEEDBACK,
        ),
        Stage18FeatureSpec(
            "local_interference_estimate",
            "local_resource_conflict_context",
            FEATURE_SOURCE_LOCAL_LINK_ESTIMATE,
        ),
        Stage18FeatureSpec(
            "recent_local_resource_rejection_count",
            "local_resource_conflict_context",
            FEATURE_SOURCE_LOCAL_PROJECTION_FEEDBACK,
        ),
        Stage18FeatureSpec(
            "neighbor_role_summary",
            "local_resource_conflict_context",
            FEATURE_SOURCE_LOCAL_MESSAGE,
        ),
        Stage18FeatureSpec(
            "estimated_link_success_probability",
            "local_communication_estimate",
            FEATURE_SOURCE_LOCAL_LINK_ESTIMATE,
        ),
        Stage18FeatureSpec(
            "estimated_deadline_delivery_probability",
            "local_communication_estimate",
            FEATURE_SOURCE_LOCAL_LINK_ESTIMATE,
        ),
        Stage18FeatureSpec(
            "estimated_p2p_latency",
            "local_communication_estimate",
            FEATURE_SOURCE_LOCAL_LINK_ESTIMATE,
        ),
        Stage18FeatureSpec(
            "estimated_p2p_energy",
            "local_communication_estimate",
            FEATURE_SOURCE_LOCAL_LINK_ESTIMATE,
        ),
        Stage18FeatureSpec("estimated_sinr", "local_communication_estimate", FEATURE_SOURCE_LOCAL_LINK_ESTIMATE),
        Stage18FeatureSpec(
            "estimated_los_nlos",
            "local_communication_estimate",
            FEATURE_SOURCE_LOCAL_LINK_ESTIMATE,
        ),
        Stage18FeatureSpec(
            "estimated_required_transmission_time",
            "local_communication_estimate",
            FEATURE_SOURCE_LOCAL_LINK_ESTIMATE,
        ),
        Stage18FeatureSpec(
            "estimated_retransmission_attempts",
            "local_communication_estimate",
            FEATURE_SOURCE_LOCAL_LINK_ESTIMATE,
        ),
        Stage18FeatureSpec(
            "recent_message_success_rate_local",
            "local_message_history",
            FEATURE_SOURCE_LOCAL_MESSAGE,
        ),
        Stage18FeatureSpec(
            "recent_retry_count_local",
            "local_message_history",
            FEATURE_SOURCE_LOCAL_HISTORY,
        ),
        Stage18FeatureSpec(
            "recent_neighbor_response_summary",
            "local_message_history",
            FEATURE_SOURCE_LOCAL_MESSAGE,
        ),
        Stage18FeatureSpec(
            "local_history_embedding_fields",
            "local_message_history",
            FEATURE_SOURCE_LOCAL_HISTORY,
        ),
        Stage18FeatureSpec(
            "time_since_last_successful_local_delivery",
            "local_message_history",
            FEATURE_SOURCE_LOCAL_HISTORY,
        ),
    )
}


def stage18_feature_schema() -> dict[str, object]:
    """Return the declared Stage 18 feature schema."""

    return {
        "schema_id": STAGE18_FEATURE_SCHEMA_ID,
        "features": {
            name: spec.to_dict()
            for name, spec in sorted(STAGE18_ACTOR_SAFE_FEATURE_SPECS.items())
        },
        "forbidden_fields": sorted(STAGE18_FORBIDDEN_ACTOR_FIELDS),
    }


def build_stage18_actor_safe_feature_rows(
    row: LearningEvidenceRow,
    *,
    previous_selected_edges: Iterable[str] = (),
) -> tuple[dict[str, object], ...]:
    """Build edge-level actor-safe feature rows for one evidence row."""

    selected = set(row.selected_edges)
    previous_selected = set(previous_selected_edges)
    output: list[dict[str, object]] = []
    for actor_row in row.actor_safe_rows:
        _reject_forbidden_actor_keys(actor_row, context="source actor row")
        agent_id = str(actor_row["agent_id"])
        current_incident_edges = _incident_edges(actor_row, selected)
        previous_incident_edges = _incident_edges(actor_row, previous_selected)
        current_selected_neighbors = _selected_neighbors(actor_row, selected)
        previous_selected_neighbors = _selected_neighbors(actor_row, previous_selected)
        rejection_histogram = _rejection_histogram(actor_row.get("local_history", {}))
        for neighbor in actor_row["local_neighbor_observations"]:
            payload = _neighbor_payload(neighbor)
            edge_id = str(payload["edge_id"])
            neighbor_kind = str(payload["neighbor_kind"])
            edge_active_current = edge_id in selected
            edge_active_prev = edge_id in previous_selected
            channel_slot = _local_channel_slot(edge_id)
            local_channel_occupancy = sum(
                1 for active_edge in current_incident_edges if _local_channel_slot(active_edge) == channel_slot
            )
            tx_budget = _budget_for_kind(str(actor_row["agent_kind"]), DEFAULT_TX_BUDGET_BY_KIND)
            neighbor_rx_capacity = _budget_for_kind(neighbor_kind, DEFAULT_RX_CAPACITY_BY_KIND)
            tx_used = len(current_incident_edges)
            success_probability = float(payload["link_success_probability"])
            latency_s = float(payload["estimated_link_latency_s"])
            energy_j = float(payload["estimated_link_energy_j"])
            feature_row = {
                "feature_schema_id": STAGE18_FEATURE_SCHEMA_ID,
                "agent_id": agent_id,
                "agent_kind": str(actor_row["agent_kind"]),
                "time_step": int(actor_row["time_step"]),
                "local_position_m": tuple(float(value) for value in actor_row["local_position_m"]),
                "local_neighbor_observations": (payload,),
                "local_messages": _message_summaries(actor_row.get("local_messages", ())),
                "local_history": _local_history_summary(actor_row.get("local_history", {})),
                "edge_id": edge_id,
                "directed_edge_id": f"{agent_id}->{payload['neighbor_id']}",
                "neighbor_id": str(payload["neighbor_id"]),
                "neighbor_kind": neighbor_kind,
                "edge_active_prev": edge_active_prev,
                "edge_active_current_local": edge_active_current,
                "local_outgoing_degree_prev": len(previous_incident_edges),
                "local_incoming_degree_estimate": tx_used,
                "local_selected_edge_count": tx_used,
                "last_action_proposed": _history_value(actor_row, "last_action_proposed", "none"),
                "last_action_accepted": _history_value(actor_row, "last_action_accepted", "none"),
                "last_projection_rejection_reason": _history_value(
                    actor_row,
                    "last_projection_rejection_reason",
                    "none",
                ),
                "recent_rejection_reason_histogram": rejection_histogram,
                "previous_selected_neighbor_summary": tuple(sorted(previous_selected_neighbors)),
                "tx_budget_used": tx_used,
                "tx_budget_remaining": max(0, tx_budget - tx_used),
                "rx_capacity_estimate_for_neighbor": max(
                    0,
                    neighbor_rx_capacity - int(edge_active_current),
                ),
                "channel_slot_available": local_channel_occupancy < tx_budget,
                "local_channel_slot_occupancy": local_channel_occupancy,
                "local_conflict_group_occupancy": local_channel_occupancy,
                "local_interference_estimate": _local_interference_estimate(
                    local_channel_occupancy=local_channel_occupancy,
                    local_candidate_count=len(tuple(actor_row["local_neighbor_observations"])),
                    link_energy_j=energy_j,
                ),
                "recent_local_resource_rejection_count": int(
                    sum(rejection_histogram.values())
                ),
                "neighbor_role_summary": {
                    "neighbor_kind": neighbor_kind,
                    "estimated_rx_capacity": neighbor_rx_capacity,
                },
                "estimated_link_success_probability": success_probability,
                "estimated_deadline_delivery_probability": success_probability,
                "estimated_p2p_latency": latency_s,
                "estimated_p2p_energy": energy_j,
                "estimated_sinr": _success_probability_to_sinr_proxy(success_probability),
                "estimated_los_nlos": _los_proxy(float(payload["distance_3d_m"])),
                "estimated_required_transmission_time": latency_s,
                "estimated_retransmission_attempts": _expected_attempts(success_probability),
                "recent_message_success_rate_local": _message_success_rate(
                    actor_row.get("local_messages", ())
                ),
                "recent_retry_count_local": int(
                    _history_value(actor_row, "recent_retry_count_local", 0)
                ),
                "recent_neighbor_response_summary": _neighbor_response_summary(
                    actor_row.get("local_messages", ()),
                    str(payload["neighbor_id"]),
                ),
                "local_history_embedding_fields": _local_history_embedding_fields(
                    actor_row.get("local_history", {}),
                    time_step=int(actor_row["time_step"]),
                ),
                "time_since_last_successful_local_delivery": float(
                    _history_value(
                        actor_row,
                        "time_since_last_successful_local_delivery",
                        int(actor_row["time_step"]),
                    )
                ),
                "current_selected_neighbor_count": len(current_selected_neighbors),
            }
            validate_stage18_actor_safe_feature_row(feature_row)
            output.append(feature_row)
    return tuple(output)


def validate_stage18_actor_safe_feature_row(row: Mapping[str, object]) -> None:
    """Reject global, objective, oracle, future, or critic fields."""

    if row.get("feature_schema_id") != STAGE18_FEATURE_SCHEMA_ID:
        raise Stage18ActorFeatureViolation("missing Stage 18 feature schema id")
    _reject_forbidden_actor_keys(row, context="stage18 actor feature row")
    for feature_name in STAGE18_ACTOR_SAFE_FEATURE_SPECS:
        if feature_name not in row:
            raise Stage18ActorFeatureViolation(f"missing actor-safe feature: {feature_name}")


def stage18_actor_signature_key(row: Mapping[str, object]) -> tuple[object, ...]:
    """Build a deterministic key for rebuilt actor-edge signatures."""

    validate_stage18_actor_safe_feature_row(row)
    neighbor = tuple(row["local_neighbor_observations"])[0]
    return (
        ("agent_id", row["agent_id"]),
        ("agent_kind", row["agent_kind"]),
        ("time_step", row["time_step"]),
        ("edge_id", row["edge_id"]),
        ("neighbor_id", row["neighbor_id"]),
        ("neighbor_kind", row["neighbor_kind"]),
        ("local_position_m", tuple(row["local_position_m"])),
        ("neighbor_link", _stable_mapping(neighbor)),
        ("edge_active_prev", row["edge_active_prev"]),
        ("edge_active_current_local", row["edge_active_current_local"]),
        ("local_outgoing_degree_prev", row["local_outgoing_degree_prev"]),
        ("local_selected_edge_count", row["local_selected_edge_count"]),
        ("tx_budget_remaining", row["tx_budget_remaining"]),
        ("local_channel_slot_occupancy", row["local_channel_slot_occupancy"]),
        ("local_conflict_group_occupancy", row["local_conflict_group_occupancy"]),
        ("previous_selected_neighbor_summary", tuple(row["previous_selected_neighbor_summary"])),
        ("recent_local_resource_rejection_count", row["recent_local_resource_rejection_count"]),
    )


def actor_safe_view_has_no_forbidden_fields(rows: Iterable[Mapping[str, object]]) -> bool:
    try:
        for row in rows:
            validate_stage18_actor_safe_feature_row(row)
    except Stage18ActorFeatureViolation:
        return False
    return True


def _reject_forbidden_actor_keys(value: object, *, context: str) -> None:
    keys: set[str] = set()
    _collect_mapping_keys(value, keys)
    forbidden = sorted(keys & STAGE18_FORBIDDEN_ACTOR_FIELDS)
    if forbidden:
        raise Stage18ActorFeatureViolation(
            f"{context} contains forbidden actor fields: {forbidden}"
        )


def _collect_mapping_keys(value: object, output: set[str]) -> None:
    if isinstance(value, Mapping):
        output.update(str(key) for key in value)
        for item in value.values():
            _collect_mapping_keys(item, output)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _collect_mapping_keys(item, output)


def _incident_edges(actor_row: Mapping[str, object], selected_edges: set[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            str(_neighbor_payload(neighbor)["edge_id"])
            for neighbor in actor_row["local_neighbor_observations"]
            if str(_neighbor_payload(neighbor)["edge_id"]) in selected_edges
        )
    )


def _selected_neighbors(actor_row: Mapping[str, object], selected_edges: set[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            str(_neighbor_payload(neighbor)["neighbor_id"])
            for neighbor in actor_row["local_neighbor_observations"]
            if str(_neighbor_payload(neighbor)["edge_id"]) in selected_edges
        )
    )


def _budget_for_kind(kind: str, budgets: Mapping[str, int]) -> int:
    return int(budgets.get(kind, 2))


def _local_channel_slot(edge_id: str) -> int:
    return sum(ord(char) for char in edge_id) % 3


def _local_interference_estimate(
    *,
    local_channel_occupancy: int,
    local_candidate_count: int,
    link_energy_j: float,
) -> float:
    denominator = max(1, local_candidate_count)
    return round(float(local_channel_occupancy) / denominator + min(1.0, link_energy_j), 6)


def _success_probability_to_sinr_proxy(probability: float) -> float:
    clipped = min(0.999999, max(0.000001, probability))
    return round(clipped / (1.0 - clipped), 6)


def _los_proxy(distance_m: float) -> str:
    return "los_estimate" if distance_m <= 100.0 else "nlos_estimate"


def _expected_attempts(probability: float) -> float:
    clipped = min(1.0, max(0.000001, probability))
    return round(1.0 / clipped, 6)


def _history_value(
    actor_row: Mapping[str, object],
    key: str,
    default: object,
) -> object:
    history = actor_row.get("local_history", {})
    if isinstance(history, Mapping):
        return history.get(key, default)
    return default


def _rejection_histogram(history: object) -> dict[str, int]:
    if isinstance(history, Mapping):
        raw = history.get("recent_rejection_reason_histogram", {})
        if isinstance(raw, Mapping):
            return {str(key): int(value) for key, value in raw.items()}
    return {"none": 0}


def _message_summaries(messages: object) -> tuple[dict[str, object], ...]:
    summaries: list[dict[str, object]] = []
    for message in messages if isinstance(messages, (tuple, list)) else ():
        payload = getattr(message, "payload", {})
        summaries.append(
            {
                "sender_id": str(getattr(message, "sender_id", "unknown")),
                "message_type": str(getattr(message, "message_type", "unknown")),
                "payload_keys": tuple(sorted(str(key) for key in dict(payload))),
            }
        )
    return tuple(summaries)


def _message_success_rate(messages: object) -> float:
    if not isinstance(messages, (tuple, list)) or not messages:
        return 0.0
    successes = 0
    total = 0
    for message in messages:
        payload = getattr(message, "payload", {})
        if isinstance(payload, Mapping) and "local_delivery_success" in payload:
            total += 1
            successes += int(bool(payload["local_delivery_success"]))
    return round(successes / total, 6) if total else 0.0


def _neighbor_response_summary(messages: object, neighbor_id: str) -> dict[str, object]:
    if not isinstance(messages, (tuple, list)):
        return {"neighbor_id": neighbor_id, "response_count": 0, "success_rate": 0.0}
    responses = [
        message
        for message in messages
        if str(getattr(message, "sender_id", "")) == neighbor_id
    ]
    return {
        "neighbor_id": neighbor_id,
        "response_count": len(responses),
        "success_rate": _message_success_rate(tuple(responses)),
    }


def _local_history_summary(history: object) -> dict[str, object]:
    if not isinstance(history, Mapping):
        return {}
    allowed = {
        "last_action_proposed",
        "last_action_accepted",
        "last_projection_rejection_reason",
        "recent_rejection_reason_histogram",
        "recent_retry_count_local",
        "time_since_last_successful_local_delivery",
    }
    return {str(key): value for key, value in history.items() if str(key) in allowed}


def _local_history_embedding_fields(history: object, *, time_step: int) -> dict[str, float]:
    if not isinstance(history, Mapping):
        return {
            "history_present": 0.0,
            "recent_retry_count_local": 0.0,
            "time_step": float(time_step),
        }
    retry_count = history.get("recent_retry_count_local", 0)
    time_since_success = history.get("time_since_last_successful_local_delivery", time_step)
    values = {
        "history_present": 1.0 if history else 0.0,
        "recent_retry_count_local": float(retry_count),
        "time_since_last_successful_local_delivery": float(time_since_success),
        "time_step": float(time_step),
    }
    for key, value in values.items():
        if not isfinite(value):
            raise Stage18ActorFeatureViolation(f"non-finite local history feature: {key}")
    return values


def _stable_mapping(value: object) -> tuple[tuple[str, object], ...]:
    if not isinstance(value, Mapping):
        return (("value", repr(value)),)
    return tuple(sorted((str(key), _stable_value(item)) for key, item in value.items()))


def _stable_value(value: object) -> object:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, Mapping):
        return _stable_mapping(value)
    if isinstance(value, (tuple, list)):
        return tuple(_stable_value(item) for item in value)
    return value

