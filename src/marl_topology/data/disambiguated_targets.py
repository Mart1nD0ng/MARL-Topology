"""Stage 18 actor-target rebuild.

Actor targets produced here are scorer targets: soft utility and pairwise
ranking. Raw global counterfactual deltas are preserved only for critic-side
or diagnostic consumers.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .actor_feature_rebuild import stage18_actor_signature_key


TARGET_ROLE_ACTOR_SOFT = "actor_soft_utility"
TARGET_ROLE_ACTOR_RANKING = "actor_pairwise_ranking"
TARGET_ROLE_HARD_DIAGNOSTIC = "hard_label_diagnostic_only"
TARGET_ROLE_CRITIC_ONLY = "critic_only"
TARGET_ROLE_DIAGNOSTICS_ONLY = "diagnostics_only"


class Stage18TargetViolation(ValueError):
    """Raised when rebuilt targets violate Stage 18 boundaries."""


@dataclass(frozen=True, slots=True)
class ActorSoftUtilityTarget:
    edge_id: str
    directed_edge_id: str
    agent_id: str
    time_step: int
    actor_edge_utility_target: float
    actor_edge_utility_confidence: float
    actor_target_source: str
    actor_training_role: str
    actor_target_ambiguity_level: str
    actor_target_allowed_for_hard_supervision: bool
    ambiguity_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 0.0 <= self.actor_edge_utility_target <= 1.0:
            raise Stage18TargetViolation("actor utility target must be in [0, 1]")
        if not 0.0 <= self.actor_edge_utility_confidence <= 1.0:
            raise Stage18TargetViolation("actor utility confidence must be in [0, 1]")
        if self.actor_training_role != "soft_distillation_only":
            raise Stage18TargetViolation("Stage 18 actor utility must be soft-distillation only")

    def to_dict(self) -> dict[str, object]:
        return {
            "edge_id": self.edge_id,
            "directed_edge_id": self.directed_edge_id,
            "agent_id": self.agent_id,
            "time_step": self.time_step,
            "actor_edge_utility_target": self.actor_edge_utility_target,
            "actor_edge_utility_confidence": self.actor_edge_utility_confidence,
            "actor_target_source": self.actor_target_source,
            "actor_training_role": self.actor_training_role,
            "actor_target_ambiguity_level": self.actor_target_ambiguity_level,
            "actor_target_allowed_for_hard_supervision": (
                self.actor_target_allowed_for_hard_supervision
            ),
            "ambiguity_reasons": list(self.ambiguity_reasons),
            "target_role": TARGET_ROLE_ACTOR_SOFT,
        }


@dataclass(frozen=True, slots=True)
class PairwiseRankingTarget:
    ranking_pair_id: str
    agent_id: str
    time_step: int
    preferred_edge_id: str
    less_preferred_edge_id: str
    preference_margin: float
    ranking_confidence: float
    ranking_source: str

    def __post_init__(self) -> None:
        if self.preferred_edge_id == self.less_preferred_edge_id:
            raise Stage18TargetViolation("ranking pair must compare distinct edges")
        if self.preference_margin <= 0:
            raise Stage18TargetViolation("ranking preference margin must be positive")
        if not 0.0 <= self.ranking_confidence <= 1.0:
            raise Stage18TargetViolation("ranking confidence must be in [0, 1]")

    def to_dict(self) -> dict[str, object]:
        return {
            "ranking_pair_id": self.ranking_pair_id,
            "agent_id": self.agent_id,
            "time_step": self.time_step,
            "preferred_edge_id": self.preferred_edge_id,
            "less_preferred_edge_id": self.less_preferred_edge_id,
            "preference_margin": self.preference_margin,
            "ranking_confidence": self.ranking_confidence,
            "ranking_source": self.ranking_source,
            "target_role": TARGET_ROLE_ACTOR_RANKING,
        }


@dataclass(frozen=True, slots=True)
class HardLabelDiagnostic:
    edge_id: str
    directed_edge_id: str
    agent_id: str
    hard_label: str
    hard_label_allowed_for_actor_training: bool
    hard_label_conflict_free: bool
    hard_label_source: str
    hard_label_confidence: float
    hard_label_rejection_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "edge_id": self.edge_id,
            "directed_edge_id": self.directed_edge_id,
            "agent_id": self.agent_id,
            "hard_label": self.hard_label,
            "hard_label_allowed_for_actor_training": (
                self.hard_label_allowed_for_actor_training
            ),
            "hard_label_conflict_free": self.hard_label_conflict_free,
            "hard_label_source": self.hard_label_source,
            "hard_label_confidence": self.hard_label_confidence,
            "hard_label_rejection_reason": self.hard_label_rejection_reason,
            "target_role": TARGET_ROLE_HARD_DIAGNOSTIC,
        }


def build_actor_target_views(
    *,
    actor_safe_rows: Iterable[Mapping[str, object]],
    raw_targets: Iterable[Mapping[str, object]],
    ambiguity_by_signature: Mapping[tuple[object, ...], tuple[str, ...]],
    hard_label_allowed_by_signature: Mapping[tuple[object, ...], bool],
    topology_label_role: str,
) -> dict[str, object]:
    """Build actor soft, ranking, and hard-label diagnostic targets."""

    target_by_edge = _counterfactual_targets(raw_targets)
    soft_targets: list[ActorSoftUtilityTarget] = []
    hard_diagnostics: list[HardLabelDiagnostic] = []
    for actor_row in actor_safe_rows:
        edge_id = str(actor_row["edge_id"])
        if edge_id not in target_by_edge:
            continue
        target = target_by_edge[edge_id]
        signature_key = stage18_actor_signature_key(actor_row)
        ambiguity_reasons = tuple(ambiguity_by_signature.get(signature_key, ()))
        hard_conflict_free = bool(hard_label_allowed_by_signature.get(signature_key, False))
        hard_allowed = hard_conflict_free and not ambiguity_reasons
        soft_target = _build_soft_target(
            actor_row=actor_row,
            target=target,
            ambiguity_reasons=ambiguity_reasons,
            topology_label_role=topology_label_role,
        )
        soft_targets.append(soft_target)
        hard_diagnostics.append(
            HardLabelDiagnostic(
                edge_id=edge_id,
                directed_edge_id=str(actor_row["directed_edge_id"]),
                agent_id=str(actor_row["agent_id"]),
                hard_label=_hard_label_from_target(target),
                hard_label_allowed_for_actor_training=hard_allowed,
                hard_label_conflict_free=hard_conflict_free,
                hard_label_source=str(target.get("action_type", "unknown")),
                hard_label_confidence=soft_target.actor_edge_utility_confidence if hard_allowed else 0.0,
                hard_label_rejection_reason=None
                if hard_allowed
                else _hard_label_rejection_reason(ambiguity_reasons),
            )
        )

    ranking_targets = _build_pairwise_targets(soft_targets)
    return {
        "actor_soft_utility_targets": tuple(target.to_dict() for target in soft_targets),
        "pairwise_ranking_targets": tuple(target.to_dict() for target in ranking_targets),
        "hard_label_diagnostics": tuple(target.to_dict() for target in hard_diagnostics),
        "target_schema_id": "stage18_disambiguated_actor_target_schema_v1",
    }


def build_critic_target_view(
    *,
    raw_targets: Iterable[Mapping[str, object]],
    selected_edges: Iterable[str],
    topology_label_role: str,
) -> dict[str, object]:
    """Preserve global counterfactual targets as critic-only diagnostics."""

    selected = set(selected_edges)
    critic_targets: list[dict[str, object]] = []
    for target in raw_targets:
        edge_id = str(target.get("edge_id", ""))
        critic_targets.append(
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
                "delta_surrogate_diagnostic": dict(
                    target.get("delta_reward_surrogate_diagnostic", {})
                ),
                "oracle_edge_used": topology_label_role == "oracle_training_diagnostic_only"
                and edge_id in selected,
                "oracle_topology_membership": topology_label_role
                == "oracle_training_diagnostic_only",
                "target_role": TARGET_ROLE_CRITIC_ONLY,
            }
        )
    return {
        "edge_delta_targets": tuple(critic_targets),
        "critic_target_schema_id": "stage18_critic_only_global_target_schema_v1",
        "target_role": TARGET_ROLE_CRITIC_ONLY,
    }


def actor_target_view_has_no_global_delta_fields(view: Mapping[str, object]) -> bool:
    text = repr(view)
    forbidden = (
        "delta_consensus_success_probability",
        "delta_latency",
        "delta_energy",
        "delta_feasibility",
        "oracle_topology_membership",
        "consensus_success_probability",
    )
    return not any(term in text for term in forbidden)


def _build_soft_target(
    *,
    actor_row: Mapping[str, object],
    target: Mapping[str, object],
    ambiguity_reasons: tuple[str, ...],
    topology_label_role: str,
) -> ActorSoftUtilityTarget:
    local_score = _local_edge_quality(actor_row)
    distilled_score = _distilled_counterfactual_score(target)
    utility = _clamp01(0.7 * local_score + 0.3 * distilled_score)
    confidence = _confidence(
        ambiguity_reasons=ambiguity_reasons,
        topology_label_role=topology_label_role,
        target=target,
    )
    return ActorSoftUtilityTarget(
        edge_id=str(actor_row["edge_id"]),
        directed_edge_id=str(actor_row["directed_edge_id"]),
        agent_id=str(actor_row["agent_id"]),
        time_step=int(actor_row["time_step"]),
        actor_edge_utility_target=round(utility, 6),
        actor_edge_utility_confidence=round(confidence, 6),
        actor_target_source="distilled_global_counterfactual",
        actor_training_role="soft_distillation_only",
        actor_target_ambiguity_level=_ambiguity_level(ambiguity_reasons, confidence),
        actor_target_allowed_for_hard_supervision=False,
        ambiguity_reasons=ambiguity_reasons,
    )


def _local_edge_quality(row: Mapping[str, object]) -> float:
    success = float(row["estimated_link_success_probability"])
    latency = min(1.0, float(row["estimated_p2p_latency"]) / 0.02)
    energy = min(1.0, float(row["estimated_p2p_energy"]) / 1.0)
    resource = min(1.0, float(row["tx_budget_remaining"]) / max(1.0, float(row["tx_budget_used"]) + 1.0))
    return _clamp01(0.55 * success + 0.15 * (1.0 - latency) + 0.15 * (1.0 - energy) + 0.15 * resource)


def _distilled_counterfactual_score(target: Mapping[str, object]) -> float:
    delta_probability = float(target.get("delta_consensus_success_probability", 0.0))
    delta_latency = float(target.get("delta_latency", 0.0))
    delta_energy = float(target.get("delta_energy", 0.0))
    delta_feasibility = float(target.get("delta_feasibility", 0.0))
    score = 0.5 + 0.35 * delta_probability + 0.10 * delta_feasibility
    score -= 0.025 * _positive(delta_latency / 0.02)
    score -= 0.025 * _positive(delta_energy)
    return _clamp01(score)


def _confidence(
    *,
    ambiguity_reasons: tuple[str, ...],
    topology_label_role: str,
    target: Mapping[str, object],
) -> float:
    confidence = 0.85
    if ambiguity_reasons:
        confidence *= 0.35
    if topology_label_role == "oracle_training_diagnostic_only":
        confidence *= 0.5
    if str(target.get("action_type", "")) == "keep_edge":
        confidence *= 0.8
    if int(target.get("delta_feasibility", 0)) != 0:
        confidence *= 0.9
    return _clamp01(confidence)


def _ambiguity_level(ambiguity_reasons: tuple[str, ...], confidence: float) -> str:
    if not ambiguity_reasons:
        return "low"
    if confidence < 0.5:
        return "high"
    return "medium"


def _build_pairwise_targets(
    soft_targets: Iterable[ActorSoftUtilityTarget],
) -> tuple[PairwiseRankingTarget, ...]:
    by_agent_time: dict[tuple[str, int], list[ActorSoftUtilityTarget]] = {}
    for target in soft_targets:
        by_agent_time.setdefault((target.agent_id, target.time_step), []).append(target)
    pairs: list[PairwiseRankingTarget] = []
    for (agent_id, time_step), targets in sorted(by_agent_time.items()):
        ranked = sorted(
            targets,
            key=lambda item: (-item.actor_edge_utility_target, item.edge_id),
        )
        for left_index, preferred in enumerate(ranked):
            for less_preferred in ranked[left_index + 1 :]:
                margin = preferred.actor_edge_utility_target - less_preferred.actor_edge_utility_target
                if margin <= 0.02:
                    continue
                confidence = min(
                    preferred.actor_edge_utility_confidence,
                    less_preferred.actor_edge_utility_confidence,
                    margin,
                )
                if confidence <= 0.0:
                    continue
                pairs.append(
                    PairwiseRankingTarget(
                        ranking_pair_id=f"{agent_id}:t{time_step}:{preferred.edge_id}>{less_preferred.edge_id}",
                        agent_id=agent_id,
                        time_step=time_step,
                        preferred_edge_id=preferred.edge_id,
                        less_preferred_edge_id=less_preferred.edge_id,
                        preference_margin=round(margin, 6),
                        ranking_confidence=round(confidence, 6),
                        ranking_source="stage18_soft_utility_ordering",
                    )
                )
    return tuple(pairs)


def _hard_label_from_target(target: Mapping[str, object]) -> str:
    action_type = str(target.get("action_type", "unknown"))
    if action_type in {"add_edge", "remove_edge", "keep_edge"}:
        return action_type
    return "unknown"


def _hard_label_rejection_reason(ambiguity_reasons: tuple[str, ...]) -> str:
    if ambiguity_reasons:
        return "signature_conflict_or_global_dependency"
    return "hard_label_not_primary_actor_target"


def _counterfactual_targets(
    raw_targets: Iterable[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    output: dict[str, Mapping[str, object]] = {}
    for target in raw_targets:
        action_type = str(target.get("action_type", ""))
        if action_type == "keep_edge":
            continue
        output[str(target.get("edge_id", ""))] = target
    return output


def _positive(value: float) -> float:
    return max(0.0, value)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))
