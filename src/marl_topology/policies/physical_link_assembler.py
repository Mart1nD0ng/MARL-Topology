"""Physical-link activation assembler for Stage 22 selected semantics."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite

from .assembler_diagnostics import AssemblerDiagnostics, RejectionReason
from .edge_scores import EdgeScoreRecord
from .topology_assembler import (
    DEPLOYMENT_ASSEMBLER_FORBIDDEN_FIELDS,
    AssemblerConfig,
    AssemblerInputViolation,
    CandidateEdgeConstraint,
)


PHYSICAL_LINK_ASSEMBLER_ID = "stage22_physical_link_conflict_aware_greedy_max_endpoint"


@dataclass(frozen=True, slots=True)
class PhysicalLinkScore:
    physical_edge_id: str
    endpoint_directed_edge_ids: tuple[str, ...]
    score: float
    probability: float | None
    aggregation_rule: str = "max_endpoint_score"
    score_source: str = "actor_endpoint_physical_link_score"
    time_step: int = 0

    def __post_init__(self) -> None:
        if not self.physical_edge_id:
            raise ValueError("physical_edge_id must be declared")
        if not self.endpoint_directed_edge_ids:
            raise ValueError("endpoint_directed_edge_ids must be non-empty")
        if not isfinite(float(self.score)):
            raise ValueError("score must be finite")
        if self.probability is not None and not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be in [0, 1] when provided")
        if self.time_step < 0:
            raise ValueError("time_step must be nonnegative")

    def to_payload(self) -> dict[str, object]:
        return {
            "physical_edge_id": self.physical_edge_id,
            "endpoint_directed_edge_ids": list(self.endpoint_directed_edge_ids),
            "score": self.score,
            "probability": self.probability,
            "aggregation_rule": self.aggregation_rule,
            "score_source": self.score_source,
            "time_step": self.time_step,
        }


@dataclass(frozen=True, slots=True)
class PhysicalLinkAssembly:
    selected_physical_edges: tuple[str, ...]
    rejected_physical_edges: tuple[str, ...]
    rejection_reasons: Mapping[str, tuple[RejectionReason, ...]]
    pre_projection_physical_edge_count: int
    post_projection_physical_edge_count: int
    assembler_id: str
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.post_projection_physical_edge_count != len(self.selected_physical_edges):
            raise ValueError("post projection count must match selected physical edges")
        if len(set(self.selected_physical_edges)) != len(self.selected_physical_edges):
            raise ValueError("selected physical edges must be unique")
        object.__setattr__(self, "selected_physical_edges", tuple(sorted(self.selected_physical_edges)))
        object.__setattr__(self, "rejected_physical_edges", tuple(sorted(self.rejected_physical_edges)))
        object.__setattr__(
            self,
            "rejection_reasons",
            {edge_id: tuple(reasons) for edge_id, reasons in self.rejection_reasons.items()},
        )
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


class PhysicalLinkConflictAwareAssembler:
    """Project endpoint-local proposals into unique physical links."""

    baseline_only = False
    recommended_for_deployment = True

    def __init__(self, config: AssemblerConfig | None = None) -> None:
        self.config = config or AssemblerConfig(
            assembler_id=PHYSICAL_LINK_ASSEMBLER_ID,
            mode="physical_link_conflict_aware_greedy",
            deterministic=True,
        )

    def assemble(
        self,
        physical_scores: Iterable[PhysicalLinkScore],
        candidate_constraints: Iterable[CandidateEdgeConstraint],
        metadata: Mapping[str, object] | None = None,
    ) -> PhysicalLinkAssembly:
        _reject_forbidden_metadata(metadata)
        scores = tuple(physical_scores)
        constraints_by_physical = _constraints_by_physical(candidate_constraints)
        selected: list[str] = []
        rejected: dict[str, list[RejectionReason]] = {}
        endpoint_usage: dict[str, float] = defaultdict(float)
        used_channels: set[str] = set()
        used_conflict_groups: set[str] = set()

        for score in sorted(scores, key=lambda item: (-item.score, item.physical_edge_id)):
            physical_edge_id = score.physical_edge_id
            constraints = constraints_by_physical.get(physical_edge_id, ())
            reasons: list[RejectionReason] = []
            if not constraints:
                reasons.append(RejectionReason.INVALID_CANDIDATE)
            if physical_edge_id in selected:
                reasons.append(RejectionReason.DUPLICATE_EDGE)
            if any(not constraint.valid_candidate for constraint in constraints):
                reasons.append(RejectionReason.INVALID_CANDIDATE)
            if any(not constraint.role_allowed for constraint in constraints):
                reasons.append(RejectionReason.ROLE_FORBIDDEN)

            if not reasons:
                endpoints = _physical_endpoints(physical_edge_id)
                if any(
                    endpoint_usage[node_id] + 1.0
                    > _physical_endpoint_capacity(self.config, node_id)
                    for node_id in endpoints
                ):
                    reasons.append(RejectionReason.TX_BUDGET_EXCEEDED)
                channels = {constraint.channel_slot for constraint in constraints if constraint.channel_slot}
                groups = {constraint.conflict_group for constraint in constraints if constraint.conflict_group}
                if channels & used_channels:
                    reasons.append(RejectionReason.CHANNEL_CONFLICT)
                if groups & used_conflict_groups:
                    reasons.append(RejectionReason.INTERFERENCE_CONFLICT)

            if reasons:
                rejected[physical_edge_id] = reasons
                continue

            selected.append(physical_edge_id)
            endpoints = _physical_endpoints(physical_edge_id)
            for node_id in endpoints:
                endpoint_usage[node_id] += 1.0
            for constraint in constraints:
                if constraint.channel_slot:
                    used_channels.add(constraint.channel_slot)
                if constraint.conflict_group:
                    used_conflict_groups.add(constraint.conflict_group)

        return _build_physical_assembly(
            selected=selected,
            rejected=rejected,
            pre_count=len(scores),
            config=self.config,
        )


def aggregate_endpoint_scores_to_physical_links(
    edge_scores: Iterable[EdgeScoreRecord],
    *,
    aggregation_rule: str = "max_endpoint_score",
) -> tuple[PhysicalLinkScore, ...]:
    """Map i->j and j->i endpoint proposals to one physical-link score."""

    if aggregation_rule != "max_endpoint_score":
        raise ValueError("Stage 22 selected aggregation_rule=max_endpoint_score")
    by_edge: dict[str, list[EdgeScoreRecord]] = defaultdict(list)
    for score in edge_scores:
        by_edge[score.edge_id].append(score)
    physical_scores: list[PhysicalLinkScore] = []
    for edge_id, records in sorted(by_edge.items()):
        chosen = max(records, key=lambda item: (item.score, item.directed_edge_id))
        probabilities = [record.probability for record in records if record.probability is not None]
        physical_scores.append(
            PhysicalLinkScore(
                physical_edge_id=edge_id,
                endpoint_directed_edge_ids=tuple(sorted(record.directed_edge_id for record in records)),
                score=chosen.score,
                probability=max(probabilities) if probabilities else chosen.probability,
                aggregation_rule=aggregation_rule,
                score_source=f"physical_link_max_endpoint:{chosen.score_source}",
                time_step=chosen.time_step,
            )
        )
    return tuple(physical_scores)


def _reject_forbidden_metadata(metadata: Mapping[str, object] | None) -> None:
    if not metadata:
        return
    forbidden = sorted(set(metadata) & DEPLOYMENT_ASSEMBLER_FORBIDDEN_FIELDS)
    if forbidden:
        raise AssemblerInputViolation(f"forbidden assembler metadata fields: {forbidden}")


def _constraints_by_physical(
    constraints: Iterable[CandidateEdgeConstraint],
) -> dict[str, tuple[CandidateEdgeConstraint, ...]]:
    grouped: dict[str, list[CandidateEdgeConstraint]] = defaultdict(list)
    for constraint in constraints:
        grouped[constraint.edge_id].append(constraint)
    return {edge_id: tuple(items) for edge_id, items in grouped.items()}


def _capacity_for(
    capacity: Mapping[str, float] | float | None,
    node_id: str,
) -> float:
    if capacity is None:
        return float("inf")
    if isinstance(capacity, Mapping):
        if node_id in capacity:
            return float(capacity[node_id])
        if "default" in capacity:
            return float(capacity["default"])
        return float("inf")
    return float(capacity)


def _physical_endpoint_capacity(config: AssemblerConfig, node_id: str) -> float:
    return max(
        _capacity_for(config.tx_capacity, node_id),
        _capacity_for(config.rx_capacity, node_id),
    )


def _physical_endpoints(physical_edge_id: str) -> tuple[str, str]:
    if "--" not in physical_edge_id:
        raise ValueError("physical_edge_id must use canonical a--b form")
    left, right = physical_edge_id.split("--", 1)
    if not left or not right or left == right:
        raise ValueError("physical edge endpoints must be distinct")
    return left, right


def _build_physical_assembly(
    *,
    selected: list[str],
    rejected: dict[str, list[RejectionReason]],
    pre_count: int,
    config: AssemblerConfig,
) -> PhysicalLinkAssembly:
    selected_tuple = tuple(sorted(set(selected)))
    rejected_tuple = tuple(sorted(rejected))
    reason_counts = Counter(
        reason.value for reasons in rejected.values() for reason in reasons
    )
    diagnostics = AssemblerDiagnostics(
        assembler_id=config.assembler_id,
        mode=config.mode,
        baseline_only=False,
        recommended_for_deployment=True,
        pre_projection_edge_count=pre_count,
        post_projection_edge_count=len(selected_tuple),
        rejected_count=len(rejected_tuple),
        rejection_reason_counts=dict(reason_counts),
        notes=(
            "Stage 22 selected physical-link deployment assembler",
            "endpoint directed proposals aggregate with max_endpoint_score",
            "final topology has unique physical edges",
        ),
        oracle_used=False,
        objective_used=False,
        reward_signal_used=False,
    )
    return PhysicalLinkAssembly(
        selected_physical_edges=selected_tuple,
        rejected_physical_edges=rejected_tuple,
        rejection_reasons=rejected,
        pre_projection_physical_edge_count=pre_count,
        post_projection_physical_edge_count=len(selected_tuple),
        assembler_id=config.assembler_id,
        diagnostics=diagnostics.to_payload(),
    )
