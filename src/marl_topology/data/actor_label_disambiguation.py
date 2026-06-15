"""Actor-local observation versus edge-label disambiguation diagnostics.

Stage 17 uses this module to explain when identical actor-safe local
observations map to contradictory edge labels. It is an analysis-only module:
no model, parameter-update routine, training loop, checkpoint, or artifact
writer lives here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from math import isclose
from typing import Any

from marl_topology.env import ACTOR_ALLOWED_FIELDS, ACTOR_FORBIDDEN_FIELDS

from .learning_evidence import ACTOR_VIEW_FORBIDDEN_FIELDS, LearningEvidenceDataset
from .learning_evidence_stage16 import build_stage16_learning_evidence_dataset


STAGE17_STAGE_ID = "stage_17_actor_observability_and_label_disambiguation"
STAGE17_VERDICT = "stage17_actor_observability_diagnosed_training_still_blocked"
STAGE17_RECOMMENDED_NEXT_TASK = (
    "stage_18_evidence_rebuild_with_disambiguated_features_or_targets"
)
SIGNATURE_SCHEMA_ID = "actor_safe_edge_observation_signature_v1"
DEFAULT_ROUND_DIGITS = 6
DEFAULT_NEAR_ZERO = 1e-9


class ActorLabelDisambiguationViolation(ValueError):
    """Raised when Stage 17 diagnostics would cross the actor boundary."""


class ContradictionCause(str, Enum):
    """Supported suspected causes for contradictory actor-local labels."""

    MISSING_LOCAL_RESOURCE_CONTEXT = "missing_local_resource_context"
    MISSING_NEIGHBOR_ROLE_OR_LINK_CONTEXT = "missing_neighbor_role_or_link_context"
    MISSING_PREVIOUS_TOPOLOGY_OR_HISTORY = "missing_previous_topology_or_history"
    TARGET_DEPENDS_ON_GLOBAL_CONTEXT = "target_depends_on_global_context"
    HARD_LABEL_SHOULD_BE_SOFT_OR_RANKED = "hard_label_should_be_soft_or_ranked"
    ORACLE_LABEL_NOT_ACTOR_OBSERVABLE = "oracle_label_not_actor_observable"
    TRUE_DEC_POMDP_AMBIGUITY = "true_dec_pomdp_ambiguity"
    POSSIBLE_TARGET_BUG = "possible_target_bug"
    UNKNOWN = "unknown"


SUPPORTED_CONTRADICTION_CAUSES = tuple(cause.value for cause in ContradictionCause)


@dataclass(frozen=True, slots=True)
class ActorObservationSignature:
    """Hashable actor-safe signature for one agent-edge observation."""

    schema_id: str
    key: tuple[object, ...]
    summary: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "key": _jsonable(self.key),
            "summary": _jsonable(self.summary),
        }


@dataclass(frozen=True, slots=True)
class EdgeLabelSample:
    """One target sample attached to an actor-safe edge observation."""

    signature: ActorObservationSignature
    agent_id: str
    edge_id: str
    neighbor_id: str
    topology_id: str
    topology_name: str
    scenario_id: str
    counterfactual_action: str
    recommended_action: str
    delta_consensus_success_probability: float
    delta_latency: float
    delta_energy: float
    delta_feasibility: int
    consensus_direction: str
    latency_direction: str
    energy_direction: str
    feasibility_direction: str
    selected_in_baseline: bool
    topology_label_role: str
    topology_family: str
    target_invariant_violation: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "edge_id": self.edge_id,
            "neighbor_id": self.neighbor_id,
            "topology_id": self.topology_id,
            "topology_name": self.topology_name,
            "scenario_id": self.scenario_id,
            "counterfactual_action": self.counterfactual_action,
            "recommended_action": self.recommended_action,
            "delta_consensus_success_probability": self.delta_consensus_success_probability,
            "delta_latency": self.delta_latency,
            "delta_energy": self.delta_energy,
            "delta_feasibility": self.delta_feasibility,
            "consensus_direction": self.consensus_direction,
            "latency_direction": self.latency_direction,
            "energy_direction": self.energy_direction,
            "feasibility_direction": self.feasibility_direction,
            "selected_in_baseline": self.selected_in_baseline,
            "topology_label_role": self.topology_label_role,
            "topology_family": self.topology_family,
            "target_invariant_violation": self.target_invariant_violation,
        }


@dataclass(frozen=True, slots=True)
class ContradictionCluster:
    """Contradictory labels for one actor-safe observation signature."""

    signature: ActorObservationSignature
    affected_agent: str
    affected_edge: str
    sample_count: int
    label_distribution: Mapping[str, Mapping[str, int]]
    metric_delta_distribution: Mapping[str, Mapping[str, object]]
    scenario_topology_ids: tuple[str, ...]
    suspected_causes: tuple[str, ...]
    suggested_fixes: tuple[str, ...]
    samples: tuple[EdgeLabelSample, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "observation_signature_summary": _jsonable(self.signature.summary),
            "affected_agent": self.affected_agent,
            "affected_edge": self.affected_edge,
            "number_of_samples": self.sample_count,
            "label_distribution": _jsonable(self.label_distribution),
            "metric_delta_distribution": _jsonable(self.metric_delta_distribution),
            "scenario_topology_ids": list(self.scenario_topology_ids),
            "suspected_cause": list(self.suspected_causes),
            "suggested_fixes": list(self.suggested_fixes),
            "samples": [sample.to_dict() for sample in self.samples],
        }


@dataclass(frozen=True, slots=True)
class ActorLabelDisambiguationReport:
    """Stage 17 actor observability and label-disambiguation report."""

    dataset_id: str
    signature_schema_id: str
    signature_fields_used: tuple[str, ...]
    sample_count: int
    signature_count: int
    contradiction_cluster_count: int
    contradictory_sample_count: int
    contradiction_rate_by_signature: float
    cause_counts: Mapping[str, int]
    contradiction_clusters: tuple[ContradictionCluster, ...]
    actor_local_observations_sufficient_for_current_hard_labels: bool
    primary_conclusion: str
    next_stage_readiness: Mapping[str, object]
    recommendations_by_cause: Mapping[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": STAGE17_STAGE_ID,
            "verdict": STAGE17_VERDICT,
            "dataset_id": self.dataset_id,
            "signature_schema_id": self.signature_schema_id,
            "signature_fields_used": list(self.signature_fields_used),
            "sample_count": self.sample_count,
            "signature_count": self.signature_count,
            "contradiction_cluster_count": self.contradiction_cluster_count,
            "contradictory_sample_count": self.contradictory_sample_count,
            "contradiction_rate_by_signature": self.contradiction_rate_by_signature,
            "cause_counts": dict(self.cause_counts),
            "contradiction_clusters": [
                cluster.to_dict() for cluster in self.contradiction_clusters
            ],
            "actor_local_observations_sufficient_for_current_hard_labels": (
                self.actor_local_observations_sufficient_for_current_hard_labels
            ),
            "primary_conclusion": self.primary_conclusion,
            "next_stage_readiness": dict(self.next_stage_readiness),
            "recommendations_by_cause": dict(self.recommendations_by_cause),
            "training_execution_allowed": False,
            "stage11_to_stage15_rerun_allowed": False,
            "ppo_mappo_allowed": False,
            "coma_allowed": False,
            "transformer_allowed": False,
            "checkpoint_creation_allowed": False,
            "artifact_written": False,
            "recommended_next_task": STAGE17_RECOMMENDED_NEXT_TASK,
        }


def build_actor_observation_signature(
    actor_row: Mapping[str, object],
    neighbor: object,
    *,
    include_time_step: bool = True,
    round_digits: int = DEFAULT_ROUND_DIGITS,
) -> ActorObservationSignature:
    """Build a hashable signature from actor-safe fields only."""

    _validate_actor_safe_row(actor_row)
    neighbor_payload = _neighbor_payload(neighbor)
    local_messages = _normalize_value(actor_row.get("local_messages", ()), round_digits)
    local_history = _normalize_value(actor_row.get("local_history", {}), round_digits)
    local_position = _round_sequence(actor_row.get("local_position_m", ()), round_digits)
    key_parts: list[object] = [
        ("agent_id", str(actor_row["agent_id"])),
        ("agent_kind", str(actor_row["agent_kind"])),
        ("local_position_m", local_position),
        ("neighbor", _normalize_value(neighbor_payload, round_digits)),
        ("local_messages", local_messages),
        ("local_history", local_history),
    ]
    if include_time_step:
        key_parts.insert(2, ("time_step", int(actor_row["time_step"])))

    summary = {
        "agent_id": str(actor_row["agent_id"]),
        "agent_kind": str(actor_row["agent_kind"]),
        "time_step": int(actor_row["time_step"]) if include_time_step else "excluded",
        "local_position_m": local_position,
        "edge_id": str(neighbor_payload["edge_id"]),
        "neighbor_id": str(neighbor_payload["neighbor_id"]),
        "neighbor_kind": str(neighbor_payload["neighbor_kind"]),
        "distance_3d_m": round(float(neighbor_payload["distance_3d_m"]), round_digits),
        "link_success_probability": round(
            float(neighbor_payload["link_success_probability"]),
            round_digits,
        ),
        "estimated_link_latency_s": round(
            float(neighbor_payload["estimated_link_latency_s"]),
            round_digits,
        ),
        "estimated_link_energy_j": round(
            float(neighbor_payload["estimated_link_energy_j"]),
            round_digits,
        ),
        "local_message_count": len(tuple(actor_row.get("local_messages", ()))),
        "local_history_keys": tuple(sorted(str(key) for key in dict(actor_row.get("local_history", {})))),
    }
    return ActorObservationSignature(
        schema_id=SIGNATURE_SCHEMA_ID,
        key=tuple(key_parts),
        summary=summary,
    )


def analyze_actor_label_disambiguation(
    dataset: LearningEvidenceDataset,
    *,
    include_time_step: bool = True,
    round_digits: int = DEFAULT_ROUND_DIGITS,
    near_zero: float = DEFAULT_NEAR_ZERO,
) -> ActorLabelDisambiguationReport:
    """Analyze whether actor-safe local observations disambiguate labels."""

    samples = _collect_edge_label_samples(
        dataset,
        include_time_step=include_time_step,
        round_digits=round_digits,
        near_zero=near_zero,
    )
    samples_by_signature: dict[tuple[object, ...], list[EdgeLabelSample]] = defaultdict(list)
    signatures: dict[tuple[object, ...], ActorObservationSignature] = {}
    for sample in samples:
        samples_by_signature[sample.signature.key].append(sample)
        signatures[sample.signature.key] = sample.signature

    clusters: list[ContradictionCluster] = []
    for signature_key, cluster_samples in samples_by_signature.items():
        if not _has_contradiction(cluster_samples):
            continue
        clusters.append(
            _build_cluster(
                signatures[signature_key],
                tuple(cluster_samples),
                near_zero=near_zero,
            )
        )

    cause_counter: Counter[str] = Counter(
        cause for cluster in clusters for cause in cluster.suspected_causes
    )
    contradictory_sample_count = sum(cluster.sample_count for cluster in clusters)
    contradiction_rate = (
        len(clusters) / len(samples_by_signature) if samples_by_signature else 0.0
    )
    sufficient = len(clusters) == 0
    return ActorLabelDisambiguationReport(
        dataset_id=dataset.dataset_id,
        signature_schema_id=SIGNATURE_SCHEMA_ID,
        signature_fields_used=(
            "agent_id",
            "agent_kind",
            "time_step" if include_time_step else "time_step_excluded",
            "local_position_m",
            "local_neighbor_observations",
            "local_messages",
            "local_history",
        ),
        sample_count=len(samples),
        signature_count=len(samples_by_signature),
        contradiction_cluster_count=len(clusters),
        contradictory_sample_count=contradictory_sample_count,
        contradiction_rate_by_signature=contradiction_rate,
        cause_counts=dict(sorted(cause_counter.items())),
        contradiction_clusters=tuple(
            sorted(
                clusters,
                key=lambda cluster: (-cluster.sample_count, cluster.affected_agent, cluster.affected_edge),
            )
        ),
        actor_local_observations_sufficient_for_current_hard_labels=sufficient,
        primary_conclusion=_primary_conclusion(sufficient, cause_counter),
        next_stage_readiness={
            "completion_gate_passed": True,
            "next_stage_readiness_gate_passed": sufficient,
            "stage11_to_stage15_rerun_allowed": False,
            "training_allowed": False,
            "blocker": None
            if sufficient
            else "current hard labels are not fully actor-observable from local signatures",
            "recommended_next_task": STAGE17_RECOMMENDED_NEXT_TASK,
        },
        recommendations_by_cause=_recommendations_by_cause(),
    )


def build_stage17_actor_label_disambiguation_report() -> ActorLabelDisambiguationReport:
    """Analyze the Stage 16 dataset for Stage 17 without writing artifacts."""

    dataset = build_stage16_learning_evidence_dataset()
    return analyze_actor_label_disambiguation(dataset)


def _collect_edge_label_samples(
    dataset: LearningEvidenceDataset,
    *,
    include_time_step: bool,
    round_digits: int,
    near_zero: float,
) -> tuple[EdgeLabelSample, ...]:
    samples: list[EdgeLabelSample] = []
    for row in dataset.rows:
        target_by_edge = _counterfactual_target_by_edge(row.learning_targets)
        keep_by_edge = _keep_target_by_edge(row.learning_targets)
        selected = set(row.selected_edges)
        for actor_row in row.actor_safe_rows:
            _validate_actor_safe_row(actor_row)
            for neighbor in actor_row["local_neighbor_observations"]:
                neighbor_payload = _neighbor_payload(neighbor)
                edge_id = str(neighbor_payload["edge_id"])
                if edge_id not in target_by_edge:
                    continue
                target = target_by_edge[edge_id]
                keep_target = keep_by_edge.get(edge_id)
                signature = build_actor_observation_signature(
                    actor_row,
                    neighbor,
                    include_time_step=include_time_step,
                    round_digits=round_digits,
                )
                delta_probability = float(
                    target.get("delta_consensus_success_probability", 0.0)
                )
                delta_latency = float(target.get("delta_latency", 0.0))
                delta_energy = float(target.get("delta_energy", 0.0))
                delta_feasibility = int(target.get("delta_feasibility", 0))
                sample = EdgeLabelSample(
                    signature=signature,
                    agent_id=str(actor_row["agent_id"]),
                    edge_id=edge_id,
                    neighbor_id=str(neighbor_payload["neighbor_id"]),
                    topology_id=row.topology_id,
                    topology_name=row.topology_name,
                    scenario_id=row.scenario_id,
                    counterfactual_action=str(target["action_type"]),
                    recommended_action=_recommended_action(target, near_zero=near_zero),
                    delta_consensus_success_probability=delta_probability,
                    delta_latency=delta_latency,
                    delta_energy=delta_energy,
                    delta_feasibility=delta_feasibility,
                    consensus_direction=_sign(delta_probability, near_zero=near_zero),
                    latency_direction=_sign(delta_latency, near_zero=near_zero),
                    energy_direction=_sign(delta_energy, near_zero=near_zero),
                    feasibility_direction=_int_sign(delta_feasibility),
                    selected_in_baseline=edge_id in selected,
                    topology_label_role=str(row.diagnostics.get("topology_label_role", "unknown")),
                    topology_family=str(row.diagnostics.get("topology_family", "unknown")),
                    target_invariant_violation=_target_invariant_violation(
                        edge_id=edge_id,
                        selected=edge_id in selected,
                        target=target,
                        keep_target=keep_target,
                        near_zero=near_zero,
                    ),
                )
                samples.append(sample)
    return tuple(samples)


def _counterfactual_target_by_edge(
    learning_targets: Iterable[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    by_edge: dict[str, Mapping[str, object]] = {}
    for target in learning_targets:
        action_type = str(target.get("action_type", ""))
        edge_id = str(target.get("edge_id", ""))
        if not edge_id or action_type == "keep_edge" or "->" in edge_id:
            continue
        by_edge[edge_id] = target
    return by_edge


def _keep_target_by_edge(
    learning_targets: Iterable[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    return {
        str(target.get("edge_id", "")): target
        for target in learning_targets
        if str(target.get("action_type", "")) == "keep_edge"
    }


def _recommended_action(target: Mapping[str, object], *, near_zero: float) -> str:
    action_type = str(target.get("action_type", "unknown"))
    probability = float(target.get("delta_consensus_success_probability", 0.0))
    feasibility = int(target.get("delta_feasibility", 0))
    surrogate_delta = _surrogate_component_sum_delta(target)
    if feasibility > 0 or probability > near_zero or surrogate_delta < -near_zero:
        return action_type
    return "keep_edge"


def _has_contradiction(samples: Iterable[EdgeLabelSample]) -> bool:
    sample_tuple = tuple(samples)
    if len(sample_tuple) < 2:
        return False
    fields = (
        "counterfactual_action",
        "recommended_action",
        "consensus_direction",
        "latency_direction",
        "energy_direction",
        "feasibility_direction",
    )
    for field_name in fields:
        if len({str(getattr(sample, field_name)) for sample in sample_tuple}) > 1:
            return True
    if any(sample.target_invariant_violation is not None for sample in sample_tuple):
        return True
    return False


def _build_cluster(
    signature: ActorObservationSignature,
    samples: tuple[EdgeLabelSample, ...],
    *,
    near_zero: float,
) -> ContradictionCluster:
    causes = _classify_causes(samples, signature=signature, near_zero=near_zero)
    return ContradictionCluster(
        signature=signature,
        affected_agent=str(signature.summary["agent_id"]),
        affected_edge=str(signature.summary["edge_id"]),
        sample_count=len(samples),
        label_distribution=_label_distribution(samples),
        metric_delta_distribution=_metric_delta_distribution(samples),
        scenario_topology_ids=tuple(
            sorted({f"{sample.scenario_id}:{sample.topology_id}" for sample in samples})
        ),
        suspected_causes=causes,
        suggested_fixes=tuple(_suggested_fix(cause) for cause in causes),
        samples=samples,
    )


def _classify_causes(
    samples: tuple[EdgeLabelSample, ...],
    *,
    signature: ActorObservationSignature,
    near_zero: float,
) -> tuple[str, ...]:
    causes: list[ContradictionCause] = []
    if any(sample.target_invariant_violation for sample in samples):
        causes.append(ContradictionCause.POSSIBLE_TARGET_BUG)

    counterfactual_actions = {sample.counterfactual_action for sample in samples}
    recommended_actions = {sample.recommended_action for sample in samples}
    selected_states = {sample.selected_in_baseline for sample in samples}
    if (
        {"add_edge", "remove_edge"} <= counterfactual_actions
        or len(recommended_actions) > 1
        and len(selected_states) > 1
    ):
        causes.append(ContradictionCause.MISSING_PREVIOUS_TOPOLOGY_OR_HISTORY)

    if _direction_conflicts(samples, "consensus_direction") or _direction_conflicts(
        samples,
        "feasibility_direction",
    ):
        causes.append(ContradictionCause.TARGET_DEPENDS_ON_GLOBAL_CONTEXT)

    if _direction_conflicts(samples, "latency_direction") or _direction_conflicts(
        samples,
        "energy_direction",
    ):
        causes.append(ContradictionCause.MISSING_LOCAL_RESOURCE_CONTEXT)

    if any("oracle" in sample.topology_label_role for sample in samples):
        causes.append(ContradictionCause.ORACLE_LABEL_NOT_ACTOR_OBSERVABLE)

    if (
        "unknown" in str(signature.summary.get("neighbor_kind", "unknown"))
        or signature.summary.get("link_success_probability") is None
    ):
        causes.append(ContradictionCause.MISSING_NEIGHBOR_ROLE_OR_LINK_CONTEXT)

    has_neutral = "keep_edge" in recommended_actions or any(
        sample.consensus_direction == "zero" for sample in samples
    )
    max_probability_delta = max(
        (abs(sample.delta_consensus_success_probability) for sample in samples),
        default=0.0,
    )
    if len(recommended_actions) > 1 and (has_neutral or max_probability_delta <= near_zero):
        causes.append(ContradictionCause.HARD_LABEL_SHOULD_BE_SOFT_OR_RANKED)
    elif len(recommended_actions) > 1:
        causes.append(ContradictionCause.HARD_LABEL_SHOULD_BE_SOFT_OR_RANKED)

    if ContradictionCause.TARGET_DEPENDS_ON_GLOBAL_CONTEXT in causes:
        causes.append(ContradictionCause.TRUE_DEC_POMDP_AMBIGUITY)

    if not causes:
        causes.append(ContradictionCause.UNKNOWN)
    return tuple(dict.fromkeys(cause.value for cause in causes))


def _direction_conflicts(samples: tuple[EdgeLabelSample, ...], field_name: str) -> bool:
    directions = {str(getattr(sample, field_name)) for sample in samples}
    nonzero = directions - {"zero"}
    return len(nonzero) > 1 or (bool(nonzero) and "zero" in directions)


def _label_distribution(samples: tuple[EdgeLabelSample, ...]) -> dict[str, dict[str, int]]:
    return {
        "counterfactual_action": _counter(getattr(sample, "counterfactual_action") for sample in samples),
        "recommended_action": _counter(getattr(sample, "recommended_action") for sample in samples),
        "consensus_direction": _counter(getattr(sample, "consensus_direction") for sample in samples),
        "latency_direction": _counter(getattr(sample, "latency_direction") for sample in samples),
        "energy_direction": _counter(getattr(sample, "energy_direction") for sample in samples),
        "feasibility_direction": _counter(getattr(sample, "feasibility_direction") for sample in samples),
        "selected_in_baseline": _counter(str(sample.selected_in_baseline) for sample in samples),
        "topology_name": _counter(sample.topology_name for sample in samples),
    }


def _metric_delta_distribution(samples: tuple[EdgeLabelSample, ...]) -> dict[str, dict[str, object]]:
    return {
        "delta_consensus_success_probability": _numeric_distribution(
            sample.delta_consensus_success_probability for sample in samples
        ),
        "delta_latency": _numeric_distribution(sample.delta_latency for sample in samples),
        "delta_energy": _numeric_distribution(sample.delta_energy for sample in samples),
        "delta_feasibility": _numeric_distribution(sample.delta_feasibility for sample in samples),
    }


def _numeric_distribution(values: Iterable[float | int]) -> dict[str, object]:
    vals = [float(value) for value in values]
    if not vals:
        return {"count": 0}
    return {
        "count": len(vals),
        "min": min(vals),
        "max": max(vals),
        "negative_count": sum(1 for value in vals if value < 0.0),
        "zero_count": sum(1 for value in vals if value == 0.0),
        "positive_count": sum(1 for value in vals if value > 0.0),
    }


def _target_invariant_violation(
    *,
    edge_id: str,
    selected: bool,
    target: Mapping[str, object],
    keep_target: Mapping[str, object] | None,
    near_zero: float,
) -> str | None:
    action_type = str(target.get("action_type", ""))
    if selected and action_type != "remove_edge":
        return f"selected edge {edge_id} should have remove_edge counterfactual"
    if not selected and action_type != "add_edge":
        return f"unselected edge {edge_id} should have add_edge counterfactual"
    if keep_target is None:
        return f"edge {edge_id} missing keep_edge target"
    keep_deltas = (
        float(keep_target.get("delta_consensus_success_probability", 0.0)),
        float(keep_target.get("delta_latency", 0.0)),
        float(keep_target.get("delta_energy", 0.0)),
        float(keep_target.get("delta_feasibility", 0.0)),
    )
    if not all(isclose(value, 0.0, abs_tol=near_zero) for value in keep_deltas):
        return f"edge {edge_id} keep_edge target has nonzero delta"
    return None


def _suggested_fix(cause: str) -> str:
    return _recommendations_by_cause().get(cause, "Inspect cluster manually and add a targeted sensor.")


def _recommendations_by_cause() -> dict[str, str]:
    return {
        ContradictionCause.MISSING_LOCAL_RESOURCE_CONTEXT.value: (
            "Add actor-safe local resource summaries such as local outgoing budget, "
            "local rx/tx capacity use, local conflict group occupancy, or assembler "
            "projection diagnostics from the previous local step; otherwise keep this "
            "effect critic-only."
        ),
        ContradictionCause.MISSING_NEIGHBOR_ROLE_OR_LINK_CONTEXT.value: (
            "Add declared actor-safe neighbor role, relative position, local channel "
            "slot, and link-quality summary if not already present."
        ),
        ContradictionCause.MISSING_PREVIOUS_TOPOLOGY_OR_HISTORY.value: (
            "Add actor-safe local history of previous incident selected edges or "
            "projection outcome, or transform add/remove labels into topology-state "
            "independent edge utility/ranking targets."
        ),
        ContradictionCause.TARGET_DEPENDS_ON_GLOBAL_CONTEXT.value: (
            "Do not force a hard actor label from a global counterfactual. Move the "
            "global dependency to critic-only targets or train actor on local ranking "
            "signals with centralized critic support."
        ),
        ContradictionCause.HARD_LABEL_SHOULD_BE_SOFT_OR_RANKED.value: (
            "Replace hard add/remove/keep labels with soft delta targets, pairwise "
            "ranking, or advantage-weighted edge scores."
        ),
        ContradictionCause.ORACLE_LABEL_NOT_ACTOR_OBSERVABLE.value: (
            "Keep oracle outputs diagnostic or target-only; never expose oracle label "
            "status to actor observations. Prefer critic-only handling."
        ),
        ContradictionCause.TRUE_DEC_POMDP_AMBIGUITY.value: (
            "Escalate to owner decision on the Dec-POMDP feature boundary: either "
            "permit additional local messages/history or accept critic-only credit "
            "assignment for this ambiguity."
        ),
        ContradictionCause.POSSIBLE_TARGET_BUG.value: (
            "Audit edge-delta target construction for selected-edge/action mismatch "
            "or nonzero keep_edge deltas before rebuilding evidence."
        ),
        ContradictionCause.UNKNOWN.value: (
            "Add a narrow diagnostic fixture that isolates topology state, resource "
            "state, and local link state one at a time."
        ),
    }


def _primary_conclusion(sufficient: bool, cause_counter: Counter[str]) -> str:
    if sufficient:
        return "actor-safe local observations are sufficient for the analyzed hard labels"
    dominant = cause_counter.most_common(1)[0][0] if cause_counter else "unknown"
    return (
        "actor-safe local observations are not sufficient for current hard labels; "
        f"dominant suspected cause is {dominant}"
    )


def _validate_actor_safe_row(actor_row: Mapping[str, object]) -> None:
    fields = set(actor_row)
    forbidden = sorted(fields & (ACTOR_FORBIDDEN_FIELDS | ACTOR_VIEW_FORBIDDEN_FIELDS))
    if forbidden:
        raise ActorLabelDisambiguationViolation(
            f"actor signature received forbidden fields: {forbidden}"
        )
    unknown = sorted(fields - ACTOR_ALLOWED_FIELDS)
    if unknown:
        raise ActorLabelDisambiguationViolation(
            f"actor signature received unregistered fields: {unknown}"
        )


def _neighbor_payload(neighbor: object) -> Mapping[str, object]:
    if is_dataclass(neighbor) and not isinstance(neighbor, type):
        return asdict(neighbor)
    if isinstance(neighbor, Mapping):
        return neighbor
    payload = {
        "neighbor_id": getattr(neighbor, "neighbor_id"),
        "neighbor_kind": getattr(neighbor, "neighbor_kind"),
        "edge_id": getattr(neighbor, "edge_id"),
        "distance_3d_m": getattr(neighbor, "distance_3d_m"),
        "link_success_probability": getattr(neighbor, "link_success_probability"),
        "estimated_link_latency_s": getattr(neighbor, "estimated_link_latency_s"),
        "estimated_link_energy_j": getattr(neighbor, "estimated_link_energy_j"),
    }
    return payload


def _normalize_value(value: object, round_digits: int) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _normalize_value(asdict(value), round_digits)
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _normalize_value(item, round_digits))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (tuple, list)):
        return tuple(_normalize_value(item, round_digits) for item in value)
    if isinstance(value, float):
        return round(value, round_digits)
    return value


def _round_sequence(value: object, round_digits: int) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)):
        raise ActorLabelDisambiguationViolation("local_position_m must be a sequence")
    return tuple(round(float(item), round_digits) for item in value)


def _surrogate_component_sum_delta(target: Mapping[str, object]) -> float:
    diagnostic = target.get("delta_reward_surrogate_diagnostic", {})
    if not isinstance(diagnostic, Mapping):
        return 0.0
    return float(diagnostic.get("delta_component_sum_diagnostic", 0.0))


def _sign(value: float, *, near_zero: float) -> str:
    if value > near_zero:
        return "positive"
    if value < -near_zero:
        return "negative"
    return "zero"


def _int_sign(value: int) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _counter(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
