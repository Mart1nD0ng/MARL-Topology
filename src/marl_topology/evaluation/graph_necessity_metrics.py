"""Stage 34 graph-necessity metrics.

The label is metric-derived, never assigned from family name alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence


TAU_REQUIREMENT_MIN = 0.9


class GraphNecessityMetricViolation(ValueError):
    """Raised when graph-necessity inputs or thresholds are invalid."""


@dataclass(frozen=True, slots=True)
class GraphNecessityThresholds:
    local_heuristic_gap: float = 0.05
    local_quality_ambiguity_pairs: int = 1
    bridge_sensitivity: float = 0.05
    weak_primary_sensitivity: float = 0.05
    role_sensitivity: float = 0.05
    graph_structure_rank_gap: int = 2

    def to_payload(self) -> dict[str, object]:
        return {
            "local_heuristic_gap": self.local_heuristic_gap,
            "local_quality_ambiguity_pairs": self.local_quality_ambiguity_pairs,
            "bridge_sensitivity": self.bridge_sensitivity,
            "weak_primary_sensitivity": self.weak_primary_sensitivity,
            "role_sensitivity": self.role_sensitivity,
            "graph_structure_rank_gap": self.graph_structure_rank_gap,
        }


@dataclass(frozen=True, slots=True)
class GraphNecessityMetrics:
    local_heuristic_gap: float
    local_quality_ambiguity: int
    bridge_sensitivity: float
    weak_primary_sensitivity: float
    role_sensitivity: float
    graph_structure_rank_gap: int
    mlp_hardness_diagnostic: bool
    graph_necessary: bool
    graph_necessity_confidence: str
    thresholds: GraphNecessityThresholds

    def to_payload(self) -> dict[str, object]:
        return {
            "local_heuristic_gap": self.local_heuristic_gap,
            "local_quality_ambiguity": self.local_quality_ambiguity,
            "bridge_sensitivity": self.bridge_sensitivity,
            "weak_primary_sensitivity": self.weak_primary_sensitivity,
            "role_sensitivity": self.role_sensitivity,
            "graph_structure_rank_gap": self.graph_structure_rank_gap,
            "mlp_hardness_diagnostic": self.mlp_hardness_diagnostic,
            "graph_necessary": self.graph_necessary,
            "graph_necessity_confidence": self.graph_necessity_confidence,
            "thresholds": self.thresholds.to_payload(),
            "label_source": "metric_thresholds_not_family_name",
        }


def compute_graph_necessity_from_scores(
    *,
    teacher_score: float,
    local_heuristic_score: float,
    local_quality_ambiguity: int = 0,
    bridge_sensitivity: float = 0.0,
    weak_primary_sensitivity: float = 0.0,
    role_sensitivity: float = 0.0,
    graph_structure_rank_gap: int = 0,
    thresholds: GraphNecessityThresholds | None = None,
) -> GraphNecessityMetrics:
    """Compute the Stage34 graph-necessity label from measured scores."""

    cfg = thresholds or GraphNecessityThresholds()
    values = (
        teacher_score,
        local_heuristic_score,
        float(local_quality_ambiguity),
        bridge_sensitivity,
        weak_primary_sensitivity,
        role_sensitivity,
        float(graph_structure_rank_gap),
    )
    if not all(isfinite(float(value)) for value in values):
        raise GraphNecessityMetricViolation("graph necessity metrics must be finite")
    local_gap = float(teacher_score) - float(local_heuristic_score)
    mlp_hard = local_gap > cfg.local_heuristic_gap
    positive_reasons = [
        local_gap > cfg.local_heuristic_gap,
        int(local_quality_ambiguity) >= cfg.local_quality_ambiguity_pairs,
        float(bridge_sensitivity) > cfg.bridge_sensitivity,
        float(weak_primary_sensitivity) > cfg.weak_primary_sensitivity,
        float(role_sensitivity) > cfg.role_sensitivity,
        int(graph_structure_rank_gap) >= cfg.graph_structure_rank_gap,
        mlp_hard,
    ]
    graph_necessary = any(positive_reasons)
    if abs(local_gap) <= 1e-12 and not any(positive_reasons[1:]):
        graph_necessary = False
    confidence = "high" if sum(bool(item) for item in positive_reasons) >= 2 else "low"
    if not graph_necessary:
        confidence = "none"
    return GraphNecessityMetrics(
        local_heuristic_gap=local_gap,
        local_quality_ambiguity=int(local_quality_ambiguity),
        bridge_sensitivity=float(bridge_sensitivity),
        weak_primary_sensitivity=float(weak_primary_sensitivity),
        role_sensitivity=float(role_sensitivity),
        graph_structure_rank_gap=int(graph_structure_rank_gap),
        mlp_hardness_diagnostic=mlp_hard,
        graph_necessary=graph_necessary,
        graph_necessity_confidence=confidence,
        thresholds=cfg,
    )


def local_quality_ambiguity_count(
    edge_rows: Sequence[Mapping[str, object]],
    teacher_edges: Sequence[str],
    *,
    tolerance: float = 0.02,
) -> int:
    teacher = set(str(edge) for edge in teacher_edges)
    scored = [
        (
            str(row.get("edge_id")),
            float(row.get("local_quality", row.get("link_success_probability", 0.0))),
        )
        for row in edge_rows
    ]
    ambiguity = 0
    for left_index, (left_id, left_score) in enumerate(scored):
        for right_id, right_score in scored[left_index + 1 :]:
            if abs(left_score - right_score) <= tolerance and (
                (left_id in teacher) != (right_id in teacher)
            ):
                ambiguity += 1
    return ambiguity


def graph_structure_rank_gap(
    edge_rows: Sequence[Mapping[str, object]],
    teacher_edges: Sequence[str],
) -> int:
    teacher = set(str(edge) for edge in teacher_edges)
    ranked = sorted(
        edge_rows,
        key=lambda row: float(row.get("local_quality", row.get("link_success_probability", 0.0))),
        reverse=True,
    )
    if not teacher:
        return 0
    local_top = {str(row.get("edge_id")) for row in ranked[: len(teacher)]}
    return len(teacher.symmetric_difference(local_top))


def measured_sensitivity(
    base_score: float,
    perturbed_score: float,
) -> float:
    if not isfinite(float(base_score)) or not isfinite(float(perturbed_score)):
        raise GraphNecessityMetricViolation("sensitivity scores must be finite")
    return max(0.0, float(base_score) - float(perturbed_score))
