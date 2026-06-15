"""Environment-side non-learning topology assemblers."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from math import inf, isfinite

from .assembler_diagnostics import AssemblerDiagnostics, RejectionReason
from .edge_scores import EdgeScoreRecord, make_directed_edge_id


DEPLOYMENT_ASSEMBLER_FORBIDDEN_FIELDS = frozenset(
    {
        "consensus_success_probability",
        "global_objective_value",
        "objective_value",
        "oracle_label",
        "oracle_action",
        "edge_delta_target",
        "edge_delta_targets",
        "reward_surrogate",
        "surrogate_components",
        "future_outcome",
        "future_outcomes",
        "stage4_pbft_reliability_result",
        "pbft_reliability_result",
        "latency",
        "energy",
    }
)


class AssemblerInputViolation(ValueError):
    """Raised when an assembler receives forbidden or inconsistent inputs."""


@dataclass(frozen=True, slots=True)
class CandidateEdgeConstraint:
    edge_id: str
    tx_id: str
    rx_id: str
    edge_type: str
    role_allowed: bool
    channel_slot: str | None
    conflict_group: str | None
    tx_capacity_cost: float = 1.0
    rx_capacity_cost: float = 1.0
    valid_candidate: bool = True

    def __post_init__(self) -> None:
        for field_name in ("edge_id", "tx_id", "rx_id", "edge_type"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must be declared")
        if self.tx_id == self.rx_id:
            raise ValueError("self-directed candidate edges are not valid")
        for field_name in ("tx_capacity_cost", "rx_capacity_cost"):
            value = float(getattr(self, field_name))
            if value < 0 or not isfinite(value):
                raise ValueError(f"{field_name} must be finite and nonnegative")

    @property
    def directed_edge_id(self) -> str:
        return make_directed_edge_id(self.tx_id, self.rx_id)


@dataclass(frozen=True, slots=True)
class AssemblerConfig:
    assembler_id: str
    mode: str
    max_outgoing_per_agent: int | None = None
    role_budget_config: Mapping[str, int] = field(default_factory=dict)
    tx_capacity: Mapping[str, float] | float | None = None
    rx_capacity: Mapping[str, float] | float | None = None
    conflict_policy: str = "reject_same_channel_or_conflict_group"
    allow_hysteresis: bool = False
    deterministic: bool = True

    def __post_init__(self) -> None:
        if not self.assembler_id:
            raise ValueError("assembler_id must be declared")
        if not self.mode:
            raise ValueError("mode must be declared")
        if self.max_outgoing_per_agent is not None and self.max_outgoing_per_agent < 0:
            raise ValueError("max_outgoing_per_agent must be nonnegative when set")
        object.__setattr__(self, "role_budget_config", dict(self.role_budget_config))


@dataclass(frozen=True, slots=True)
class AssembledTopology:
    selected_directed_edges: tuple[str, ...]
    rejected_edges: tuple[str, ...]
    rejection_reasons: Mapping[str, tuple[RejectionReason, ...]]
    pre_projection_edge_count: int
    post_projection_edge_count: int
    assembler_id: str
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.pre_projection_edge_count < 0 or self.post_projection_edge_count < 0:
            raise ValueError("projection counts must be nonnegative")
        if self.post_projection_edge_count != len(self.selected_directed_edges):
            raise ValueError("post_projection_edge_count must match selected edges")
        object.__setattr__(self, "selected_directed_edges", tuple(self.selected_directed_edges))
        object.__setattr__(self, "rejected_edges", tuple(self.rejected_edges))
        object.__setattr__(
            self,
            "rejection_reasons",
            {edge_id: tuple(reasons) for edge_id, reasons in self.rejection_reasons.items()},
        )
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


class ThresholdAssembler:
    baseline_only = True
    recommended_for_deployment = False

    def __init__(
        self,
        threshold: float,
        config: AssemblerConfig | None = None,
    ) -> None:
        if not isfinite(float(threshold)):
            raise ValueError("threshold must be finite")
        self.threshold = float(threshold)
        self.config = config or AssemblerConfig(
            assembler_id="threshold_baseline",
            mode="threshold_baseline",
            deterministic=True,
        )

    def assemble(
        self,
        edge_scores: Iterable[EdgeScoreRecord],
        candidate_constraints: Iterable[CandidateEdgeConstraint],
        metadata: Mapping[str, object] | None = None,
    ) -> AssembledTopology:
        _reject_forbidden_metadata(metadata)
        scores = tuple(edge_scores)
        constraints = _constraint_map(candidate_constraints)
        selected: list[str] = []
        rejected: dict[str, list[RejectionReason]] = {}
        for score in scores:
            basic_reasons = _basic_rejection_reasons(score, constraints)
            if basic_reasons:
                rejected[score.directed_edge_id] = basic_reasons
            elif score.score >= self.threshold:
                selected.append(score.directed_edge_id)
            else:
                rejected[score.directed_edge_id] = [RejectionReason.LOW_SCORE]
        return _build_topology(
            selected=selected,
            rejected=rejected,
            pre_count=len(scores),
            config=self.config,
            baseline_only=True,
            recommended_for_deployment=False,
            notes=("threshold baseline only", "may produce full or empty behavior"),
        )


class FixedTopKPerAgentAssembler:
    baseline_only = True
    recommended_for_deployment = False

    def __init__(self, k: int, config: AssemblerConfig | None = None) -> None:
        if k < 0:
            raise ValueError("k must be nonnegative")
        self.k = k
        self.config = config or AssemblerConfig(
            assembler_id="fixed_top_k_per_agent_baseline",
            mode="fixed_top_k_per_agent_baseline",
            max_outgoing_per_agent=k,
            deterministic=True,
        )

    def assemble(
        self,
        edge_scores: Iterable[EdgeScoreRecord],
        candidate_constraints: Iterable[CandidateEdgeConstraint],
        metadata: Mapping[str, object] | None = None,
    ) -> AssembledTopology:
        _reject_forbidden_metadata(metadata)
        scores = tuple(edge_scores)
        constraints = _constraint_map(candidate_constraints)
        by_agent = _scores_by_agent(scores)
        selected: list[str] = []
        rejected: dict[str, list[RejectionReason]] = {}
        for agent_id in sorted(by_agent):
            chosen_for_agent = 0
            for score in _rank_scores(by_agent[agent_id]):
                basic_reasons = _basic_rejection_reasons(score, constraints)
                if basic_reasons:
                    rejected[score.directed_edge_id] = basic_reasons
                elif chosen_for_agent < self.k:
                    selected.append(score.directed_edge_id)
                    chosen_for_agent += 1
                else:
                    rejected[score.directed_edge_id] = [RejectionReason.LOW_SCORE]
        return _build_topology(
            selected=selected,
            rejected=rejected,
            pre_count=len(scores),
            config=self.config,
            baseline_only=True,
            recommended_for_deployment=False,
            notes=("fixed top-k is a baseline only", "not the final deployment policy"),
        )


class RoleAwareAdaptiveBudgetAssembler:
    baseline_only = False
    recommended_for_deployment = False

    def __init__(self, config: AssemblerConfig | None = None) -> None:
        self.config = config or AssemblerConfig(
            assembler_id="role_aware_adaptive_budget",
            mode="role_aware_adaptive_budget",
            role_budget_config={"default": 1, "rsu": 3, "vehicle": 1},
            deterministic=True,
        )

    def assemble(
        self,
        edge_scores: Iterable[EdgeScoreRecord],
        candidate_constraints: Iterable[CandidateEdgeConstraint],
        metadata: Mapping[str, object] | None = None,
    ) -> AssembledTopology:
        _reject_forbidden_metadata(metadata)
        scores = tuple(edge_scores)
        constraints = _constraint_map(candidate_constraints)
        agent_roles = _agent_roles(metadata)
        by_agent = _scores_by_agent(scores)
        selected: list[str] = []
        rejected: dict[str, list[RejectionReason]] = {}
        for agent_id in sorted(by_agent):
            budget = _adaptive_role_budget(
                role=agent_roles.get(agent_id, "default"),
                neighbor_count=len(by_agent[agent_id]),
                config=self.config,
            )
            chosen_for_agent = 0
            for score in _rank_scores(by_agent[agent_id]):
                basic_reasons = _basic_rejection_reasons(score, constraints)
                if basic_reasons:
                    rejected[score.directed_edge_id] = basic_reasons
                elif chosen_for_agent < budget:
                    selected.append(score.directed_edge_id)
                    chosen_for_agent += 1
                else:
                    rejected[score.directed_edge_id] = [RejectionReason.PROJECTION_LIMIT]
        return _build_topology(
            selected=selected,
            rejected=rejected,
            pre_count=len(scores),
            config=self.config,
            baseline_only=False,
            recommended_for_deployment=False,
            notes=("role-dependent local budget projection", "non-learning"),
        )


class ConflictAwareGreedyAssembler:
    baseline_only = False
    recommended_for_deployment = True

    def __init__(self, config: AssemblerConfig | None = None) -> None:
        self.config = config or AssemblerConfig(
            assembler_id="conflict_aware_greedy_deployment",
            mode="conflict_aware_greedy",
            deterministic=True,
        )

    def assemble(
        self,
        edge_scores: Iterable[EdgeScoreRecord],
        candidate_constraints: Iterable[CandidateEdgeConstraint],
        metadata: Mapping[str, object] | None = None,
    ) -> AssembledTopology:
        _reject_forbidden_metadata(metadata)
        scores = tuple(edge_scores)
        constraints = _constraint_map(candidate_constraints)
        selected: list[str] = []
        rejected: dict[str, list[RejectionReason]] = {}
        seen_scores: set[str] = set()
        tx_usage: dict[str, float] = defaultdict(float)
        rx_usage: dict[str, float] = defaultdict(float)
        outgoing_counts: dict[str, int] = defaultdict(int)
        used_channels: set[str] = set()
        used_conflict_groups: set[str] = set()

        for score in _rank_scores(scores):
            reasons: list[RejectionReason] = []
            if score.directed_edge_id in seen_scores:
                reasons.append(RejectionReason.DUPLICATE_EDGE)
            seen_scores.add(score.directed_edge_id)

            constraint = constraints.get(score.directed_edge_id)
            reasons.extend(_basic_rejection_reasons(score, constraints))
            if not reasons and constraint is not None:
                if (
                    self.config.max_outgoing_per_agent is not None
                    and outgoing_counts[score.agent_id] >= self.config.max_outgoing_per_agent
                ):
                    reasons.append(RejectionReason.PROJECTION_LIMIT)
                if (
                    tx_usage[constraint.tx_id] + constraint.tx_capacity_cost
                    > _capacity_for(self.config.tx_capacity, constraint.tx_id)
                ):
                    reasons.append(RejectionReason.TX_BUDGET_EXCEEDED)
                if (
                    rx_usage[constraint.rx_id] + constraint.rx_capacity_cost
                    > _capacity_for(self.config.rx_capacity, constraint.rx_id)
                ):
                    reasons.append(RejectionReason.RX_CAPACITY_EXCEEDED)
                if constraint.channel_slot and constraint.channel_slot in used_channels:
                    reasons.append(RejectionReason.CHANNEL_CONFLICT)
                if (
                    constraint.conflict_group
                    and constraint.conflict_group in used_conflict_groups
                ):
                    reasons.append(RejectionReason.INTERFERENCE_CONFLICT)

            if reasons:
                rejected[score.directed_edge_id] = reasons
                continue

            selected.append(score.directed_edge_id)
            if constraint is not None:
                tx_usage[constraint.tx_id] += constraint.tx_capacity_cost
                rx_usage[constraint.rx_id] += constraint.rx_capacity_cost
                outgoing_counts[score.agent_id] += 1
                if constraint.channel_slot:
                    used_channels.add(constraint.channel_slot)
                if constraint.conflict_group:
                    used_conflict_groups.add(constraint.conflict_group)

        return _build_topology(
            selected=selected,
            rejected=rejected,
            pre_count=len(scores),
            config=self.config,
            baseline_only=False,
            recommended_for_deployment=True,
            notes=("Stage 8 recommended deployment assembler", "greedy score projection"),
        )


def _reject_forbidden_metadata(metadata: Mapping[str, object] | None) -> None:
    if not metadata:
        return
    forbidden = sorted(set(metadata) & DEPLOYMENT_ASSEMBLER_FORBIDDEN_FIELDS)
    if forbidden:
        raise AssemblerInputViolation(f"forbidden assembler metadata fields: {forbidden}")


def _constraint_map(
    candidate_constraints: Iterable[CandidateEdgeConstraint],
) -> dict[str, CandidateEdgeConstraint]:
    constraints: dict[str, CandidateEdgeConstraint] = {}
    for constraint in candidate_constraints:
        if constraint.directed_edge_id in constraints:
            raise AssemblerInputViolation(
                f"duplicate candidate constraint: {constraint.directed_edge_id}"
            )
        constraints[constraint.directed_edge_id] = constraint
    return constraints


def _basic_rejection_reasons(
    score: EdgeScoreRecord,
    constraints: Mapping[str, CandidateEdgeConstraint],
) -> list[RejectionReason]:
    constraint = constraints.get(score.directed_edge_id)
    if constraint is None:
        return [RejectionReason.INVALID_CANDIDATE]
    if not constraint.valid_candidate:
        return [RejectionReason.INVALID_CANDIDATE]
    if not constraint.role_allowed:
        return [RejectionReason.ROLE_FORBIDDEN]
    if constraint.edge_id != score.edge_id:
        return [RejectionReason.INVALID_CANDIDATE]
    return []


def _rank_scores(scores: Iterable[EdgeScoreRecord]) -> list[EdgeScoreRecord]:
    return sorted(
        scores,
        key=lambda score: (-score.score, score.time_step, score.directed_edge_id),
    )


def _scores_by_agent(
    scores: Iterable[EdgeScoreRecord],
) -> dict[str, list[EdgeScoreRecord]]:
    by_agent: dict[str, list[EdgeScoreRecord]] = defaultdict(list)
    for score in scores:
        by_agent[score.agent_id].append(score)
    return by_agent


def _agent_roles(metadata: Mapping[str, object] | None) -> dict[str, str]:
    if not metadata:
        return {}
    raw_roles = metadata.get("agent_roles", {})
    if not isinstance(raw_roles, Mapping):
        raise AssemblerInputViolation("agent_roles metadata must be a mapping")
    return {str(agent_id): str(role) for agent_id, role in raw_roles.items()}


def _adaptive_role_budget(
    *,
    role: str,
    neighbor_count: int,
    config: AssemblerConfig,
) -> int:
    values = config.role_budget_config
    base = int(values.get(role, values.get("default", config.max_outgoing_per_agent or 1)))
    dense_threshold = int(values.get("dense_neighbor_threshold", 4))
    dense_delta = int(values.get("dense_budget_delta", -1))
    sparse_threshold = int(values.get("sparse_neighbor_threshold", 1))
    sparse_delta = int(values.get("sparse_budget_delta", 1))
    if neighbor_count >= dense_threshold:
        base += dense_delta
    elif neighbor_count <= sparse_threshold:
        base += sparse_delta
    return max(0, base)


def _capacity_for(
    capacity: Mapping[str, float] | float | None,
    agent_id: str,
) -> float:
    if capacity is None:
        return inf
    if isinstance(capacity, Mapping):
        if agent_id in capacity:
            return float(capacity[agent_id])
        if "default" in capacity:
            return float(capacity["default"])
        return inf
    return float(capacity)


def _build_topology(
    *,
    selected: list[str],
    rejected: dict[str, list[RejectionReason]],
    pre_count: int,
    config: AssemblerConfig,
    baseline_only: bool,
    recommended_for_deployment: bool,
    notes: tuple[str, ...],
) -> AssembledTopology:
    selected_tuple = tuple(sorted(selected))
    rejected_tuple = tuple(sorted(rejected))
    reason_counts = Counter(
        reason.value for reasons in rejected.values() for reason in reasons
    )
    diagnostics = AssemblerDiagnostics(
        assembler_id=config.assembler_id,
        mode=config.mode,
        baseline_only=baseline_only,
        recommended_for_deployment=recommended_for_deployment,
        pre_projection_edge_count=pre_count,
        post_projection_edge_count=len(selected_tuple),
        rejected_count=len(rejected_tuple),
        rejection_reason_counts=dict(reason_counts),
        notes=notes,
        oracle_used=False,
        objective_used=False,
        reward_signal_used=False,
    )
    return AssembledTopology(
        selected_directed_edges=selected_tuple,
        rejected_edges=rejected_tuple,
        rejection_reasons=rejected,
        pre_projection_edge_count=pre_count,
        post_projection_edge_count=len(selected_tuple),
        assembler_id=config.assembler_id,
        diagnostics=diagnostics.to_payload(),
    )
