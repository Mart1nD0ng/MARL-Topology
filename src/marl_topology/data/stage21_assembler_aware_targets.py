"""Stage 21 assembler-aware actor target rebuild.

Actor targets in this module are local edge-priority targets. Objective-stack
counterfactuals are kept in critic-only views and are never actor inputs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite

from marl_topology.policies.edge_scores import EdgeScoreRecord
from marl_topology.policies.topology_assembler import (
    AssemblerConfig,
    AssembledTopology,
    CandidateEdgeConstraint,
    ConflictAwareGreedyAssembler,
)


STAGE21_ACTOR_TARGET_SCHEMA_ID = "stage21_assembler_aware_actor_target_schema_v1"
STAGE21_CRITIC_TARGET_SCHEMA_ID = "stage21_critic_only_objective_delta_schema_v1"
STAGE21_ASSEMBLER_ID = "stage21_conflict_aware_greedy_environment_projection"
TARGET_ROLE_ACTOR_SOFT = "actor_soft_utility"
TARGET_ROLE_ACTOR_RANKING = "actor_pairwise_ranking"
TARGET_ROLE_HARD_DIAGNOSTIC = "hard_label_diagnostic_only"
TARGET_ROLE_CRITIC_ONLY = "critic_only"

ACTOR_TARGET_FORBIDDEN_TERMS = (
    "delta_consensus_success_probability",
    "delta_latency",
    "delta_energy",
    "delta_feasibility",
    "consensus_success_probability",
    "global_topology",
    "oracle_topology_membership",
    "reward_surrogate",
    "future_outcome",
)


class Stage21TargetViolation(ValueError):
    """Raised when Stage 21 target views violate actor/critic boundaries."""


@dataclass(frozen=True, slots=True)
class Stage21TeacherProjection:
    """Assembler-projected teacher signal for one scenario/time/local view."""

    score_records: tuple[EdgeScoreRecord, ...]
    constraints: tuple[CandidateEdgeConstraint, ...]
    assembled: AssembledTopology
    teacher_selected_physical_edges: tuple[str, ...]
    teacher_source: str


def build_stage21_teacher_projection(
    actor_safe_view: Iterable[Mapping[str, object]],
    *,
    proposed_edge_ids: Iterable[str] | None = None,
    teacher_source: str = "projected_greedy_reliability_teacher",
    include_non_proposed_for_targets: bool = True,
) -> Stage21TeacherProjection:
    """Project a heuristic teacher through the deployment assembler family."""

    actor_rows = tuple(actor_safe_view)
    proposed = set(proposed_edge_ids) if proposed_edge_ids is not None else None
    constraints = candidate_constraints_from_actor_safe_view(actor_rows)
    scores = edge_score_records_from_actor_safe_view(
        actor_rows,
        score_source=f"stage21_{teacher_source}",
        proposed_edge_ids=proposed,
        include_non_proposed=include_non_proposed_for_targets,
    )
    assembler = build_stage21_conflict_aware_assembler(actor_rows)
    assembled = assembler.assemble(
        scores,
        constraints,
        metadata={"agent_roles": agent_roles_from_actor_safe_view(actor_rows)},
    )
    selected = selected_physical_edges_from_assembly(scores, assembled.selected_directed_edges)
    return Stage21TeacherProjection(
        score_records=scores,
        constraints=constraints,
        assembled=assembled,
        teacher_selected_physical_edges=selected,
        teacher_source=teacher_source,
    )


def build_stage21_actor_target_view(
    *,
    actor_safe_view: Iterable[Mapping[str, object]],
    teacher_projection: Stage21TeacherProjection,
) -> dict[str, object]:
    """Build assembler-aware soft/ranking actor targets for one row."""

    actor_rows = tuple(actor_safe_view)
    rows_by_directed = {
        str(row["directed_edge_id"]): row
        for row in actor_rows
    }
    score_by_directed = {
        score.directed_edge_id: score
        for score in teacher_projection.score_records
    }
    selected = set(teacher_projection.assembled.selected_directed_edges)
    rejected = set(teacher_projection.assembled.rejected_edges)
    rejection_reasons = {
        directed_id: tuple(reason.value for reason in reasons)
        for directed_id, reasons in teacher_projection.assembled.rejection_reasons.items()
    }

    soft_targets: list[dict[str, object]] = []
    hard_diagnostics: list[dict[str, object]] = []
    low_priority_signals: list[dict[str, object]] = []
    for directed_id, row in sorted(rows_by_directed.items()):
        score = score_by_directed.get(directed_id)
        raw_score = float(score.score) if score is not None else _teacher_raw_score(row)
        reason_tuple = rejection_reasons.get(directed_id, ())
        priority = _target_priority(
            directed_id=directed_id,
            selected=selected,
            rejected=rejected,
            rejection_reasons=reason_tuple,
            raw_score=raw_score,
            row=row,
        )
        utility = _utility_for_priority(priority, raw_score)
        confidence = _confidence_for_priority(priority, reason_tuple)
        ambiguity = _ambiguity_for_priority(priority, confidence)
        signal_reason = _low_priority_reason(priority, reason_tuple, row)
        if signal_reason is not None:
            low_priority_signals.append(
                {
                    "edge_id": str(row["edge_id"]),
                    "directed_edge_id": directed_id,
                    "agent_id": str(row["agent_id"]),
                    "reason": signal_reason,
                    "raw_teacher_score": round(raw_score, 6),
                    "target_role": "actor_low_priority_or_abstain_signal",
                }
            )
        soft_targets.append(
            {
                "edge_id": str(row["edge_id"]),
                "directed_edge_id": directed_id,
                "agent_id": str(row["agent_id"]),
                "time_step": int(row["time_step"]),
                "actor_edge_utility_target": round(utility, 6),
                "actor_edge_utility_confidence": round(confidence, 6),
                "actor_target_source": teacher_projection.teacher_source,
                "actor_training_role": "assembler_aware_soft_priority",
                "actor_target_ambiguity_level": ambiguity,
                "actor_target_allowed_for_hard_supervision": False,
                "allowed_for_actor_training": confidence > 0.0,
                "ambiguity_reasons": _ambiguity_reasons(priority, reason_tuple),
                "assembler_priority_class": priority,
                "teacher_projection_status": "selected"
                if directed_id in selected
                else "rejected"
                if directed_id in rejected
                else "not_proposed",
                "target_role": TARGET_ROLE_ACTOR_SOFT,
            }
        )
        hard_diagnostics.append(
            {
                "edge_id": str(row["edge_id"]),
                "directed_edge_id": directed_id,
                "agent_id": str(row["agent_id"]),
                "hard_label": "prefer"
                if directed_id in selected
                else "abstain_or_low_priority",
                "hard_label_allowed_for_actor_training": False,
                "hard_label_conflict_free": False,
                "hard_label_source": "stage21_teacher_projection_diagnostic_only",
                "hard_label_confidence": 0.0,
                "hard_label_rejection_reason": "hard_labels_not_actor_training_target",
                "target_role": TARGET_ROLE_HARD_DIAGNOSTIC,
            }
        )

    ranking_targets = _pairwise_ranking_targets(soft_targets)
    view = {
        "actor_soft_utility_targets": tuple(soft_targets),
        "pairwise_ranking_targets": tuple(ranking_targets),
        "hard_label_diagnostics": tuple(hard_diagnostics),
        "low_priority_abstain_signals": tuple(low_priority_signals),
        "target_distribution_summary": target_distribution_summary(soft_targets),
        "proposal_rejection_diagnostics": proposal_rejection_diagnostics(
            teacher_projection.score_records,
            teacher_projection.assembled,
        ),
        "target_schema_id": STAGE21_ACTOR_TARGET_SCHEMA_ID,
    }
    if not actor_target_view_has_no_forbidden_global_fields(view):
        raise Stage21TargetViolation("Stage 21 actor target view contains forbidden global fields")
    return view


def build_stage21_critic_target_view(
    *,
    raw_targets: Iterable[Mapping[str, object]],
    selected_edges: Iterable[str],
    per_primary_reliability: Mapping[str, float],
    objective_contract_id: str,
) -> dict[str, object]:
    """Build critic-only objective-stack counterfactual targets."""

    selected = set(selected_edges)
    edge_targets: list[dict[str, object]] = []
    for target in raw_targets:
        edge_id = str(target.get("edge_id", ""))
        edge_targets.append(
            {
                "topology_id": str(target.get("topology_id", "")),
                "edge_id": edge_id,
                "action_type": str(target.get("action_type", "")),
                "delta_consensus_success_probability": float(
                    target.get("delta_consensus_success_probability", 0.0)
                ),
                "delta_latency": float(target.get("delta_latency", 0.0)),
                "delta_energy": float(target.get("delta_energy", 0.0)),
                "delta_feasibility": int(target.get("delta_feasibility", 0)),
                "objective_surrogate_diagnostic": dict(
                    target.get("delta_reward_surrogate_diagnostic", {})
                ),
                "selected_edge_membership": edge_id in selected,
                "target_role": TARGET_ROLE_CRITIC_ONLY,
            }
        )
    return {
        "edge_delta_targets": tuple(edge_targets),
        "per_primary_reliability": dict(per_primary_reliability),
        "objective_contract_id": objective_contract_id,
        "critic_target_schema_id": STAGE21_CRITIC_TARGET_SCHEMA_ID,
        "target_role": TARGET_ROLE_CRITIC_ONLY,
    }


def actor_target_view_has_no_forbidden_global_fields(view: Mapping[str, object]) -> bool:
    text = repr(view)
    return not any(term in text for term in ACTOR_TARGET_FORBIDDEN_TERMS)


def critic_targets_are_critic_only(view: Mapping[str, object]) -> bool:
    return all(
        isinstance(target, Mapping)
        and target.get("target_role") == TARGET_ROLE_CRITIC_ONLY
        for target in tuple(view.get("edge_delta_targets", ()))
    )


def target_distribution_summary(
    soft_targets: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    targets = [float(target["actor_edge_utility_target"]) for target in soft_targets]
    confidence = [
        float(target["actor_edge_utility_confidence"])
        for target in soft_targets
    ]
    if not targets:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "variance": None,
            "high_priority_count": 0,
            "mid_priority_count": 0,
            "low_priority_count": 0,
            "confidence_mean": None,
            "uniformly_high": False,
        }
    mean = sum(targets) / len(targets)
    variance = sum((value - mean) ** 2 for value in targets) / len(targets)
    return {
        "count": len(targets),
        "min": min(targets),
        "max": max(targets),
        "mean": mean,
        "variance": variance,
        "high_priority_count": sum(value >= 0.7 for value in targets),
        "mid_priority_count": sum(0.35 <= value < 0.7 for value in targets),
        "low_priority_count": sum(value < 0.35 for value in targets),
        "confidence_mean": sum(confidence) / len(confidence) if confidence else None,
        "uniformly_high": min(targets) >= 0.7,
    }


def aggregate_target_distribution(
    target_views: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    views = tuple(target_views)
    targets = [
        target
        for view in views
        for target in tuple(view.get("actor_soft_utility_targets", ()))
    ]
    rankings = [
        target
        for view in views
        for target in tuple(view.get("pairwise_ranking_targets", ()))
    ]
    low = [
        target
        for view in views
        for target in tuple(view.get("low_priority_abstain_signals", ()))
    ]
    summary = target_distribution_summary(targets)
    summary.update(
        {
            "ranking_pair_count": len(rankings),
            "low_priority_abstain_signal_count": len(low),
            "has_high_mid_low_priority": bool(
                int(summary["high_priority_count"]) > 0
                and int(summary["mid_priority_count"]) > 0
                and int(summary["low_priority_count"]) > 0
            ),
        }
    )
    return summary


def proposal_rejection_diagnostics(
    score_records: Iterable[EdgeScoreRecord],
    assembled: AssembledTopology,
    *,
    threshold: float = 0.5,
) -> dict[str, object]:
    scores = tuple(score_records)
    if not scores:
        return {
            "proposal_count": 0,
            "top_proposal_rejection_rate": 0.0,
            "above_threshold_rejection_rate": 0.0,
            "rejection_rate_by_score_quantile": {},
            "rejection_reason_by_rank": {},
            "accepted_edge_score_distribution": _score_distribution(()),
            "rejected_edge_score_distribution": _score_distribution(()),
            "rejected_high_score_edge_count": 0,
            "accepted_low_score_edge_count": 0,
        }
    rejected = set(assembled.rejected_edges)
    accepted = set(assembled.selected_directed_edges)
    ranked = sorted(scores, key=lambda item: (-item.score, item.directed_edge_id))
    top_count = max(1, min(5, len(ranked)))
    top_rejected = sum(1 for score in ranked[:top_count] if score.directed_edge_id in rejected)
    above = [score for score in scores if score.probability >= threshold]
    above_rejected = sum(1 for score in above if score.directed_edge_id in rejected)
    accepted_scores = [score.score for score in scores if score.directed_edge_id in accepted]
    rejected_scores = [score.score for score in scores if score.directed_edge_id in rejected]
    return {
        "proposal_count": len(scores),
        "top_proposal_rejection_rate": top_rejected / top_count,
        "above_threshold_rejection_rate": above_rejected / len(above) if above else 0.0,
        "rejection_rate_by_score_quantile": _rejection_rate_by_quantile(scores, rejected),
        "rejection_reason_by_rank": _rejection_reason_by_rank(ranked, assembled),
        "accepted_edge_score_distribution": _score_distribution(accepted_scores),
        "rejected_edge_score_distribution": _score_distribution(rejected_scores),
        "rejected_high_score_edge_count": sum(
            1 for score in scores if score.probability >= threshold and score.directed_edge_id in rejected
        ),
        "accepted_low_score_edge_count": sum(
            1 for score in scores if score.probability < threshold and score.directed_edge_id in accepted
        ),
    }


def aggregate_proposal_rejection_diagnostics(
    diagnostics: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    rows = tuple(diagnostics)
    if not rows:
        return {
            "top_proposal_rejection_rate": 0.0,
            "above_threshold_rejection_rate": 0.0,
            "rejected_high_score_edge_count": 0,
            "accepted_low_score_edge_count": 0,
        }
    return {
        "top_proposal_rejection_rate": _mean(
            float(row.get("top_proposal_rejection_rate", 0.0)) for row in rows
        ),
        "above_threshold_rejection_rate": _mean(
            float(row.get("above_threshold_rejection_rate", 0.0)) for row in rows
        ),
        "rejected_high_score_edge_count": sum(
            int(row.get("rejected_high_score_edge_count", 0)) for row in rows
        ),
        "accepted_low_score_edge_count": sum(
            int(row.get("accepted_low_score_edge_count", 0)) for row in rows
        ),
    }


def edge_score_records_from_actor_safe_view(
    actor_safe_view: Iterable[Mapping[str, object]],
    *,
    score_source: str,
    proposed_edge_ids: set[str] | None = None,
    include_non_proposed: bool = True,
) -> tuple[EdgeScoreRecord, ...]:
    records: list[EdgeScoreRecord] = []
    for row in actor_safe_view:
        edge_id = str(row["edge_id"])
        proposed = proposed_edge_ids is None or edge_id in proposed_edge_ids
        if not proposed and not include_non_proposed:
            continue
        raw_score = _teacher_raw_score(row)
        if not proposed:
            raw_score -= 3.0
        records.append(
            EdgeScoreRecord(
                agent_id=str(row["agent_id"]),
                neighbor_id=str(row["neighbor_id"]),
                edge_id=edge_id,
                directed_edge_id=str(row["directed_edge_id"]),
                score=raw_score,
                probability=_logit_like_probability(raw_score),
                score_source=score_source,
                time_step=int(row["time_step"]),
            )
        )
    return tuple(records)


def candidate_constraints_from_actor_safe_view(
    actor_safe_view: Iterable[Mapping[str, object]],
) -> tuple[CandidateEdgeConstraint, ...]:
    constraints: list[CandidateEdgeConstraint] = []
    for row in actor_safe_view:
        constraints.append(
            CandidateEdgeConstraint(
                edge_id=str(row["edge_id"]),
                tx_id=str(row["agent_id"]),
                rx_id=str(row["neighbor_id"]),
                edge_type="stage21_actor_local_candidate",
                role_allowed=True,
                channel_slot=None,
                conflict_group=f"physical_edge:{row['edge_id']}",
                tx_capacity_cost=1.0,
                rx_capacity_cost=1.0,
                valid_candidate=bool(row["channel_slot_available"]),
            )
        )
    return tuple(constraints)


def build_stage21_conflict_aware_assembler(
    actor_safe_view: Iterable[Mapping[str, object]],
    *,
    assembler_id: str = STAGE21_ASSEMBLER_ID,
) -> ConflictAwareGreedyAssembler:
    actor_rows = tuple(actor_safe_view)
    return ConflictAwareGreedyAssembler(
        AssemblerConfig(
            assembler_id=assembler_id,
            mode="conflict_aware_greedy",
            tx_capacity=tx_capacity_from_actor_safe_view(actor_rows),
            rx_capacity=rx_capacity_from_actor_safe_view(actor_rows),
            deterministic=True,
        )
    )


def tx_capacity_from_actor_safe_view(
    actor_safe_view: Iterable[Mapping[str, object]],
) -> dict[str, float]:
    capacity: dict[str, float] = {}
    for row in actor_safe_view:
        agent_id = str(row["agent_id"])
        capacity[agent_id] = max(capacity.get(agent_id, 0.0), float(row["tx_budget_remaining"]))
    return capacity


def rx_capacity_from_actor_safe_view(
    actor_safe_view: Iterable[Mapping[str, object]],
) -> dict[str, float]:
    capacity: dict[str, float] = {}
    for row in actor_safe_view:
        neighbor_id = str(row["neighbor_id"])
        capacity[neighbor_id] = max(
            capacity.get(neighbor_id, 0.0),
            float(row["rx_capacity_estimate_for_neighbor"]),
        )
    return capacity


def agent_roles_from_actor_safe_view(
    actor_safe_view: Iterable[Mapping[str, object]],
) -> dict[str, str]:
    roles: dict[str, str] = {}
    for row in actor_safe_view:
        roles[str(row["agent_id"])] = str(row["agent_kind"])
        roles[str(row["neighbor_id"])] = str(row["neighbor_kind"])
    return roles


def selected_physical_edges_from_assembly(
    scores: Iterable[EdgeScoreRecord],
    selected_directed_edges: Iterable[str],
) -> tuple[str, ...]:
    by_directed = {score.directed_edge_id: score.edge_id for score in scores}
    return tuple(
        sorted(
            {
                by_directed[directed_edge_id]
                for directed_edge_id in selected_directed_edges
                if directed_edge_id in by_directed
            }
        )
    )


def _target_priority(
    *,
    directed_id: str,
    selected: set[str],
    rejected: set[str],
    rejection_reasons: tuple[str, ...],
    raw_score: float,
    row: Mapping[str, object],
) -> str:
    if directed_id in selected:
        return "high"
    if directed_id in rejected:
        if "invalid_candidate" in rejection_reasons:
            return "invalid_low"
        if any("budget" in reason or "capacity" in reason for reason in rejection_reasons):
            return "resource_reject_low"
        if any("conflict" in reason for reason in rejection_reasons):
            return "conflict_reject_low"
        if "duplicate_edge" in rejection_reasons:
            return "redundant_low"
        if "projection_limit" in rejection_reasons:
            return "projection_limit_low"
        return "rejected_low"
    if _high_cost(row):
        return "high_cost_low"
    if raw_score > 0.25:
        return "mid"
    return "low"


def _utility_for_priority(priority: str, raw_score: float) -> float:
    normalized = max(0.0, min(1.0, 0.5 + 0.2 * raw_score))
    if priority == "high":
        return max(0.75, min(0.95, 0.82 + 0.12 * normalized))
    if priority == "mid":
        return max(0.4, min(0.65, 0.45 + 0.2 * normalized))
    if priority in {"low", "rejected_low"}:
        return 0.25
    if priority == "high_cost_low":
        return 0.18
    if priority in {"resource_reject_low", "conflict_reject_low", "projection_limit_low"}:
        return 0.12
    if priority in {"invalid_low", "redundant_low"}:
        return 0.06
    return 0.2


def _confidence_for_priority(priority: str, rejection_reasons: tuple[str, ...]) -> float:
    if priority == "high":
        return 0.9
    if priority == "mid":
        return 0.65
    if rejection_reasons:
        return 0.85
    return 0.55


def _ambiguity_for_priority(priority: str, confidence: float) -> str:
    if confidence >= 0.8 and priority != "mid":
        return "low"
    if confidence >= 0.55:
        return "medium"
    return "high"


def _ambiguity_reasons(priority: str, rejection_reasons: tuple[str, ...]) -> tuple[str, ...]:
    if rejection_reasons:
        return tuple(f"assembler_{reason}" for reason in rejection_reasons)
    if priority == "mid":
        return ("local_priority_mid_confidence",)
    return ()


def _low_priority_reason(
    priority: str,
    rejection_reasons: tuple[str, ...],
    row: Mapping[str, object],
) -> str | None:
    if priority == "high":
        return None
    if rejection_reasons:
        return "|".join(rejection_reasons)
    if priority == "high_cost_low":
        return "high_cost_edge_not_selected_by_teacher_projection"
    if priority in {"low", "rejected_low"}:
        return "not_selected_by_teacher_projection"
    if priority == "mid":
        return None
    if bool(row.get("edge_active_current_local", False)):
        return "redundant_edge_under_current_local_topology"
    return priority


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
            key=lambda item: (-float(item["actor_edge_utility_target"]), str(item["edge_id"])),
        )
        for left_index, preferred in enumerate(ranked):
            for less_preferred in ranked[left_index + 1 :]:
                margin = float(preferred["actor_edge_utility_target"]) - float(
                    less_preferred["actor_edge_utility_target"]
                )
                if margin <= 0.08:
                    continue
                confidence = min(
                    float(preferred["actor_edge_utility_confidence"]),
                    float(less_preferred["actor_edge_utility_confidence"]),
                    margin,
                )
                pairs.append(
                    {
                        "ranking_pair_id": (
                            f"{agent_id}:t{time_step}:"
                            f"{preferred['edge_id']}>{less_preferred['edge_id']}"
                        ),
                        "agent_id": agent_id,
                        "time_step": time_step,
                        "preferred_edge_id": str(preferred["edge_id"]),
                        "less_preferred_edge_id": str(less_preferred["edge_id"]),
                        "preference_margin": round(margin, 6),
                        "ranking_confidence": round(confidence, 6),
                        "ranking_source": "stage21_assembler_projection_ordering",
                        "target_role": TARGET_ROLE_ACTOR_RANKING,
                    }
                )
    return tuple(pairs)


def _teacher_raw_score(row: Mapping[str, object]) -> float:
    success = float(row["estimated_link_success_probability"])
    deadline = float(row["estimated_deadline_delivery_probability"])
    latency = min(1.0, float(row["estimated_p2p_latency"]) / 0.02)
    energy = min(1.0, float(row["estimated_p2p_energy"]) / 0.01)
    resource = min(1.0, float(row["tx_budget_remaining"]) / 4.0)
    conflict = min(1.0, float(row["local_conflict_group_occupancy"]) / 4.0)
    return (
        1.8 * success
        + 0.7 * deadline
        + 0.3 * resource
        - 0.2 * latency
        - 0.2 * energy
        - 0.15 * conflict
    )


def _high_cost(row: Mapping[str, object]) -> bool:
    return float(row["estimated_p2p_energy"]) > 0.006 or float(row["estimated_p2p_latency"]) > 0.006


def _logit_like_probability(score: float) -> float:
    if not isfinite(score):
        return 0.0
    # Smooth monotone squashing without importing torch or scipy into data code.
    return max(0.0, min(1.0, 0.5 + score / (2.0 * (1.0 + abs(score)))))


def _rejection_rate_by_quantile(
    scores: tuple[EdgeScoreRecord, ...],
    rejected: set[str],
) -> dict[str, float]:
    ranked = sorted(scores, key=lambda item: item.score)
    if not ranked:
        return {}
    buckets = {
        "q1_lowest": ranked[: max(1, len(ranked) // 4)],
        "q2_mid_low": ranked[max(1, len(ranked) // 4) : max(1, len(ranked) // 2)],
        "q3_mid_high": ranked[max(1, len(ranked) // 2) : max(1, (3 * len(ranked)) // 4)],
        "q4_highest": ranked[max(1, (3 * len(ranked)) // 4) :],
    }
    return {
        name: (
            sum(1 for score in bucket if score.directed_edge_id in rejected) / len(bucket)
            if bucket
            else 0.0
        )
        for name, bucket in buckets.items()
    }


def _rejection_reason_by_rank(
    ranked: list[EdgeScoreRecord],
    assembled: AssembledTopology,
) -> dict[str, dict[str, int]]:
    buckets: dict[str, Counter[str]] = {
        "top_5": Counter(),
        "rest": Counter(),
    }
    for index, score in enumerate(ranked):
        reasons = assembled.rejection_reasons.get(score.directed_edge_id, ())
        bucket = "top_5" if index < 5 else "rest"
        for reason in reasons:
            buckets[bucket][reason.value] += 1
    return {name: dict(counter) for name, counter in buckets.items()}


def _score_distribution(values: Iterable[float]) -> dict[str, float | None]:
    items = [float(value) for value in values if isfinite(float(value))]
    if not items:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(items),
        "mean": sum(items) / len(items),
        "min": min(items),
        "max": max(items),
    }


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0
