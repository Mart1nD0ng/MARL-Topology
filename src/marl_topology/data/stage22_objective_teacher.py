"""Stage 22 objective-aware teacher and target construction."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass


STAGE22_OBJECTIVE_TEACHER_ID = "stage22_objective_aware_teacher_v2"
STAGE22_ACTOR_TARGET_SCHEMA_ID = "stage22_objective_aware_actor_target_schema_v1"
STAGE22_CRITIC_TARGET_SCHEMA_ID = "stage22_critic_only_objective_delta_schema_v1"
STAGE22_TAU_REQUIREMENT_MIN = 0.9

STAGE22_ACTOR_TARGET_FORBIDDEN_TERMS = (
    "delta_consensus_success_probability",
    "delta_latency",
    "delta_energy",
    "delta_feasibility",
    "consensus_success_probability",
    "global_topology",
    "oracle_membership",
    "objective_value",
    "reward_surrogate",
    "future_outcome",
)


@dataclass(frozen=True, slots=True)
class TeacherCandidate:
    candidate_id: str
    selected_directed_edges: tuple[str, ...]
    selected_physical_edges: tuple[str, ...]
    source: str

    @property
    def edge_count(self) -> int:
        if self.selected_physical_edges:
            return len(self.selected_physical_edges)
        return len(self.selected_directed_edges)


@dataclass(frozen=True, slots=True)
class TeacherEvaluation:
    candidate: TeacherCandidate
    consensus_success_probability: float
    latency: float
    energy: float
    feasible_under_tau_requirement: bool

    @property
    def objective_key(self) -> tuple[int, float, float, int, str]:
        return (
            0 if self.feasible_under_tau_requirement else 1,
            self.latency,
            self.energy,
            self.candidate.edge_count,
            self.candidate.candidate_id,
        )


@dataclass(frozen=True, slots=True)
class ObjectiveAwareTeacherResult:
    action_semantics_id: str
    teacher_id: str
    selected_candidate: TeacherCandidate
    evaluated_candidates: tuple[TeacherEvaluation, ...]
    search_method: str

    @property
    def teacher_feasible(self) -> bool:
        return bool(self.best_evaluation.feasible_under_tau_requirement)

    @property
    def best_evaluation(self) -> TeacherEvaluation:
        return min(self.evaluated_candidates, key=lambda item: item.objective_key)

    def to_summary(self) -> dict[str, object]:
        best = self.best_evaluation
        projected_greedy = next(
            (
                item
                for item in self.evaluated_candidates
                if "greedy_reliability" in item.candidate.candidate_id
            ),
            None,
        )
        return {
            "teacher_id": self.teacher_id,
            "action_semantics_id": self.action_semantics_id,
            "search_method": self.search_method,
            "candidate_count": len(self.evaluated_candidates),
            "selected_candidate_id": best.candidate.candidate_id,
            "teacher_feasible": best.feasible_under_tau_requirement,
            "teacher_tau_feasible": best.feasible_under_tau_requirement,
            "teacher_consensus_success_probability": best.consensus_success_probability,
            "teacher_latency": best.latency,
            "teacher_energy": best.energy,
            "teacher_selected_physical_edges": list(best.candidate.selected_physical_edges),
            "teacher_selected_directed_edges": list(best.candidate.selected_directed_edges),
            "projected_greedy_teacher_gap": _teacher_gap(best, projected_greedy),
            "target_source": STAGE22_OBJECTIVE_TEACHER_ID,
        }


def select_objective_aware_teacher(
    *,
    action_semantics_id: str,
    candidates: Iterable[TeacherCandidate],
    evaluate_candidate: Callable[[TeacherCandidate], Mapping[str, object]],
    search_method: str,
) -> ObjectiveAwareTeacherResult:
    """Choose a teacher topology using the Stage 5 objective ordering."""

    evaluations: list[TeacherEvaluation] = []
    for candidate in candidates:
        metrics = evaluate_candidate(candidate)
        probability = float(metrics["consensus_success_probability"])
        evaluations.append(
            TeacherEvaluation(
                candidate=candidate,
                consensus_success_probability=probability,
                latency=float(metrics["latency"]),
                energy=float(metrics["energy"]),
                feasible_under_tau_requirement=probability >= STAGE22_TAU_REQUIREMENT_MIN,
            )
        )
    if not evaluations:
        raise ValueError("objective-aware teacher requires candidate topologies")
    selected = min(evaluations, key=lambda item: item.objective_key).candidate
    return ObjectiveAwareTeacherResult(
        action_semantics_id=action_semantics_id,
        teacher_id=STAGE22_OBJECTIVE_TEACHER_ID,
        selected_candidate=selected,
        evaluated_candidates=tuple(sorted(evaluations, key=lambda item: item.objective_key)),
        search_method=search_method,
    )


def build_stage22_actor_target_view(
    *,
    action_semantics_id: str,
    actor_safe_view: Iterable[Mapping[str, object]],
    teacher_result: ObjectiveAwareTeacherResult,
    rejected_reasons_by_edge: Mapping[str, tuple[str, ...]],
) -> dict[str, object]:
    """Build objective-aware soft/ranking actor targets."""

    rows = tuple(actor_safe_view)
    best = teacher_result.best_evaluation
    selected_physical = set(best.candidate.selected_physical_edges)
    objective_rank = _objective_rank_by_physical_edge(teacher_result)
    soft_targets: list[dict[str, object]] = []
    low_priority: list[dict[str, object]] = []
    hard_diagnostics: list[dict[str, object]] = []

    for row in rows:
        edge_id = str(row["edge_id"])
        directed_id = str(row["directed_edge_id"])
        teacher_selected = edge_id in selected_physical
        reasons = tuple(rejected_reasons_by_edge.get(directed_id, ())) or tuple(
            rejected_reasons_by_edge.get(edge_id, ())
        )
        selected = teacher_selected and not reasons
        rank = int(objective_rank.get(edge_id, 999))
        utility, confidence, priority, reason = _target_for_edge(
            selected=selected,
            teacher_feasible=best.feasible_under_tau_requirement,
            rank=rank,
            row=row,
            rejection_reasons=reasons,
        )
        if reason is not None:
            low_priority.append(
                {
                    "edge_id": edge_id,
                    "directed_edge_id": directed_id,
                    "agent_id": str(row["agent_id"]),
                    "low_priority_reason": reason,
                    "teacher_rejection_reason": reason,
                    "target_role": "actor_low_priority_or_abstain_signal",
                }
            )
        soft_targets.append(
            {
                "edge_id": edge_id,
                "directed_edge_id": directed_id,
                "agent_id": str(row["agent_id"]),
                "time_step": int(row["time_step"]),
                "actor_edge_utility_target": utility,
                "actor_edge_utility_confidence": confidence,
                "ranking_pairs": (),
                "low_priority_reason": reason,
                "teacher_selected": teacher_selected,
                "teacher_rejection_reason": reason,
                "teacher_objective_rank": rank,
                "teacher_feasible": best.feasible_under_tau_requirement,
                "target_source": STAGE22_OBJECTIVE_TEACHER_ID,
                "target_allowed_for_actor_training": confidence > 0.0,
                "allowed_for_actor_training": confidence > 0.0,
                "actor_target_source": STAGE22_OBJECTIVE_TEACHER_ID,
                "actor_target_ambiguity_level": "low" if confidence >= 0.8 else "medium",
                "actor_training_role": "objective_aware_soft_priority",
                "assembler_priority_class": priority,
                "target_role": "actor_soft_utility",
            }
        )
        hard_diagnostics.append(
            {
                "edge_id": edge_id,
                "directed_edge_id": directed_id,
                "agent_id": str(row["agent_id"]),
                "hard_label": "teacher_selected" if selected else "teacher_not_selected",
                "hard_label_allowed_for_actor_training": False,
                "hard_label_source": "stage22_objective_teacher_diagnostic_only",
                "target_role": "hard_label_diagnostic_only",
            }
        )

    ranking_targets = _pairwise_ranking_targets(soft_targets)
    view = {
        "actor_soft_utility_targets": tuple(soft_targets),
        "pairwise_ranking_targets": tuple(ranking_targets),
        "hard_label_diagnostics": tuple(hard_diagnostics),
        "low_priority_abstain_signals": tuple(low_priority),
        "target_distribution_summary": target_distribution_summary(soft_targets),
        "teacher_summary": teacher_result.to_summary(),
        "target_schema_id": STAGE22_ACTOR_TARGET_SCHEMA_ID,
    }
    if not actor_target_view_has_no_forbidden_global_fields(view):
        raise ValueError("Stage 22 actor target view contains forbidden global fields")
    return view


def build_stage22_critic_target_view(
    *,
    action_semantics_id: str,
    teacher_result: ObjectiveAwareTeacherResult,
    edge_delta_rows: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    targets = []
    for row in edge_delta_rows:
        edge_id = str(row.get("edge_id", ""))
        targets.append(
            {
                "edge_id": edge_id,
                "action_semantics_id": action_semantics_id,
                "delta_consensus_success_probability": float(
                    row.get("delta_consensus_success_probability", 0.0)
                ),
                "delta_latency": float(row.get("delta_latency", 0.0)),
                "delta_energy": float(row.get("delta_energy", 0.0)),
                "delta_feasibility": int(row.get("delta_feasibility", 0)),
                "oracle_membership": edge_id
                in set(teacher_result.best_evaluation.candidate.selected_physical_edges),
                "objective_value": _objective_value(teacher_result.best_evaluation),
                "target_role": "critic_only",
            }
        )
    return {
        "critic_target_schema_id": STAGE22_CRITIC_TARGET_SCHEMA_ID,
        "action_semantics_id": action_semantics_id,
        "edge_delta_targets": tuple(targets),
        "teacher_summary": teacher_result.to_summary(),
        "target_role": "critic_only",
    }


def actor_target_view_has_no_forbidden_global_fields(view: Mapping[str, object]) -> bool:
    actor_only = {
        key: value
        for key, value in view.items()
        if key
        in {
            "actor_soft_utility_targets",
            "pairwise_ranking_targets",
            "hard_label_diagnostics",
            "low_priority_abstain_signals",
            "target_distribution_summary",
            "target_schema_id",
        }
    }
    text = repr(actor_only)
    return not any(term in text for term in STAGE22_ACTOR_TARGET_FORBIDDEN_TERMS)


def critic_targets_are_critic_only(view: Mapping[str, object]) -> bool:
    return all(
        isinstance(target, Mapping) and target.get("target_role") == "critic_only"
        for target in tuple(view.get("edge_delta_targets", ()))
    )


def target_distribution_summary(
    soft_targets: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    values = [float(target["actor_edge_utility_target"]) for target in soft_targets]
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "variance": None,
            "high_priority_count": 0,
            "mid_priority_count": 0,
            "low_priority_count": 0,
            "uniformly_high": False,
            "has_high_mid_low_priority": False,
        }
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    high = sum(value >= 0.7 for value in values)
    mid = sum(0.35 <= value < 0.7 for value in values)
    low = sum(value < 0.35 for value in values)
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "variance": variance,
        "high_priority_count": high,
        "mid_priority_count": mid,
        "low_priority_count": low,
        "uniformly_high": min(values) >= 0.7,
        "has_high_mid_low_priority": high > 0 and mid > 0 and low > 0,
    }


def aggregate_stage22_target_distribution(
    target_views: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    views = tuple(target_views)
    targets = [
        target
        for view in views
        for target in tuple(view.get("actor_soft_utility_targets", ()))
    ]
    summary = target_distribution_summary(targets)
    summary["ranking_pair_count"] = sum(
        len(tuple(view.get("pairwise_ranking_targets", ()))) for view in views
    )
    summary["low_priority_abstain_signal_count"] = sum(
        len(tuple(view.get("low_priority_abstain_signals", ()))) for view in views
    )
    return summary


def _target_for_edge(
    *,
    selected: bool,
    teacher_feasible: bool,
    rank: int,
    row: Mapping[str, object],
    rejection_reasons: tuple[str, ...],
) -> tuple[float, float, str, str | None]:
    if selected:
        if rank <= 2:
            return 0.93, 0.92 if teacher_feasible else 0.75, "high", None
        if rank <= 5:
            return 0.58, 0.72, "selected_mid_marginal", None
        return (
            0.28,
            0.72,
            "selected_low_marginal",
            "selected_by_teacher_but_low_marginal_priority",
        )
    if rejection_reasons:
        return 0.08, 0.9, "rejected_low", "|".join(rejection_reasons)
    if not bool(row.get("channel_slot_available", True)):
        return 0.06, 0.9, "invalid_low", "invalid_candidate_or_channel_unavailable"
    if rank <= 2:
        return 0.55, 0.65, "mid", None
    if float(row.get("estimated_link_energy_j", 0.0)) > 0.05:
        return 0.16, 0.8, "high_cost_low", "high_cost_edge_not_selected_by_teacher"
    return 0.24, 0.65, "low", "not_selected_by_objective_teacher"


def _objective_rank_by_physical_edge(
    teacher_result: ObjectiveAwareTeacherResult,
) -> dict[str, int]:
    best_rank: dict[str, int] = {}
    single_edge_evaluations = [
        evaluation
        for evaluation in teacher_result.evaluated_candidates
        if len(evaluation.candidate.selected_physical_edges) == 1
    ]
    source = single_edge_evaluations or list(teacher_result.evaluated_candidates)
    for rank, evaluation in enumerate(source, start=1):
        for edge_id in evaluation.candidate.selected_physical_edges:
            best_rank[edge_id] = min(best_rank.get(edge_id, rank), rank)
    return best_rank


def _pairwise_ranking_targets(
    soft_targets: Iterable[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    by_agent_time: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    for target in soft_targets:
        by_agent_time[(str(target["agent_id"]), int(target["time_step"]))].append(target)
    pairs: list[dict[str, object]] = []
    for (agent_id, time_step), targets in sorted(by_agent_time.items()):
        ranked = sorted(
            targets,
            key=lambda item: (
                -float(item["actor_edge_utility_target"]),
                str(item["edge_id"]),
            ),
        )
        for preferred, less_preferred in zip(ranked, reversed(ranked)):
            margin = float(preferred["actor_edge_utility_target"]) - float(
                less_preferred["actor_edge_utility_target"]
            )
            if margin <= 0.15:
                continue
            pairs.append(
                {
                    "agent_id": agent_id,
                    "time_step": time_step,
                    "preferred_edge_id": str(preferred["edge_id"]),
                    "less_preferred_edge_id": str(less_preferred["edge_id"]),
                    "preference_margin": round(margin, 6),
                    "ranking_confidence": min(
                        float(preferred["actor_edge_utility_confidence"]),
                        float(less_preferred["actor_edge_utility_confidence"]),
                    ),
                    "ranking_source": STAGE22_OBJECTIVE_TEACHER_ID,
                }
            )
    return tuple(pairs)


def _objective_value(evaluation: TeacherEvaluation) -> float:
    feasibility_penalty = 0.0 if evaluation.feasible_under_tau_requirement else 1.0
    return feasibility_penalty + evaluation.latency + evaluation.energy


def _teacher_gap(
    best: TeacherEvaluation,
    projected_greedy: TeacherEvaluation | None,
) -> dict[str, object]:
    if projected_greedy is None:
        return {"available": False}
    return {
        "available": True,
        "delta_consensus_success_probability": best.consensus_success_probability
        - projected_greedy.consensus_success_probability,
        "delta_latency": best.latency - projected_greedy.latency,
        "delta_energy": best.energy - projected_greedy.energy,
        "best_feasible": best.feasible_under_tau_requirement,
        "projected_greedy_feasible": projected_greedy.feasible_under_tau_requirement,
    }
