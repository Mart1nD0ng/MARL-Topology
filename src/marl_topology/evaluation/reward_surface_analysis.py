"""Stage 25 reward-surface and objective-alignment diagnostics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite

from marl_topology.objectives import (
    SurrogateSignalConfig,
    SurrogateSignalInput,
    evaluate_reward_surrogate,
)


STAGE25_REWARD_SURFACE_ANALYSIS_ID = "stage25_reward_surface_objective_alignment_v1"
STAGE25_OBJECTIVE_ORDERING = "feasibility_first_then_latency_then_energy"


@dataclass(frozen=True, slots=True)
class ObjectiveSurfaceRecord:
    row_id: str
    policy_label: str
    scenario_id: str
    selected_edge_count: int
    candidate_edge_count: int
    consensus_success_probability: float
    latency: float
    energy: float
    feasible_under_tau: bool
    surrogate_score: float
    reliability_violation_component: float
    latency_component: float
    energy_component: float
    dominated: bool
    non_dominated: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "row_id": self.row_id,
            "policy_label": self.policy_label,
            "scenario_id": self.scenario_id,
            "selected_edge_count": self.selected_edge_count,
            "candidate_edge_count": self.candidate_edge_count,
            "consensus_success_probability": self.consensus_success_probability,
            "latency": self.latency,
            "energy": self.energy,
            "feasible_under_tau": self.feasible_under_tau,
            "surrogate_reward": self.surrogate_score,
            "reliability_violation_component": self.reliability_violation_component,
            "latency_component": self.latency_component,
            "energy_component": self.energy_component,
            "dominated": self.dominated,
            "non_dominated": self.non_dominated,
        }


def build_reward_surface_analysis(
    policy_records: Iterable[Mapping[str, object]],
    *,
    reward_config: SurrogateSignalConfig,
    tau_requirement_min: float,
) -> dict[str, object]:
    """Evaluate reward/objective alignment for fixed topology records."""

    rows = _surface_rows(
        tuple(policy_records),
        reward_config=reward_config,
        tau_requirement_min=tau_requirement_min,
    )
    checks = _alignment_checks(rows, tau_requirement_min=tau_requirement_min)
    reward_rank = sorted(rows, key=lambda row: (-row.surrogate_score, row.row_id))
    objective_rank = sorted(rows, key=_objective_key)
    return {
        "analysis_id": STAGE25_REWARD_SURFACE_ANALYSIS_ID,
        "objective_ordering": STAGE25_OBJECTIVE_ORDERING,
        "tau_requirement_min": tau_requirement_min,
        "reward_config_id": reward_config.config_id,
        "reward_weights": {
            "reliability_weight": reward_config.reliability_weight,
            "latency_weight": reward_config.latency_weight,
            "energy_weight": reward_config.energy_weight,
            "tau": reward_config.tau,
        },
        "row_count": len(rows),
        "rows": [row.to_dict() for row in rows],
        "reward_rank_order": [row.row_id for row in reward_rank],
        "objective_rank_order": [row.row_id for row in objective_rank],
        "checks": checks,
        "alignment_passed": all(bool(check["passed"]) for check in checks.values()),
        "weight_tuning_performed": False,
    }


def _surface_rows(
    records: tuple[Mapping[str, object], ...],
    *,
    reward_config: SurrogateSignalConfig,
    tau_requirement_min: float,
) -> tuple[ObjectiveSurfaceRecord, ...]:
    provisional: list[dict[str, object]] = []
    for index, record in enumerate(records):
        probability = _finite_float(record, "consensus_success_probability")
        latency = _finite_float(record, "latency")
        energy = _finite_float(record, "energy")
        signal = evaluate_reward_surrogate(
            SurrogateSignalInput(
                consensus_success_probability=probability,
                latency=latency,
                energy=energy,
                topology_diagnostics=(
                    record.get("topology_diagnostics")
                    if isinstance(record.get("topology_diagnostics"), Mapping)
                    else {}
                ),
            ),
            reward_config,
        )
        provisional.append(
            {
                "row_id": str(record.get("row_id", f"surface_row_{index}")),
                "policy_label": str(record.get("policy_label", "unknown_policy")),
                "scenario_id": str(record.get("scenario_id", "unknown_scenario")),
                "selected_edge_count": int(record.get("selected_edge_count", 0)),
                "candidate_edge_count": int(record.get("candidate_edge_count", 0)),
                "consensus_success_probability": probability,
                "latency": latency,
                "energy": energy,
                "feasible_under_tau": probability >= tau_requirement_min,
                "surrogate_score": float(signal.training_signal_value),
                "reliability_violation_component": float(signal.reliability_penalty),
                "latency_component": float(signal.latency_penalty),
                "energy_component": float(signal.energy_penalty),
            }
        )
    dominated_flags = _dominated_flags(provisional)
    return tuple(
        ObjectiveSurfaceRecord(
            **item,
            dominated=dominated_flags[index],
            non_dominated=not dominated_flags[index],
        )
        for index, item in enumerate(provisional)
    )


def _alignment_checks(
    rows: tuple[ObjectiveSurfaceRecord, ...],
    *,
    tau_requirement_min: float,
) -> dict[str, dict[str, object]]:
    dominated_infeasible_misranks = []
    for feasible in rows:
        if not feasible.feasible_under_tau:
            continue
        for candidate in rows:
            if feasible.scenario_id != candidate.scenario_id:
                continue
            if candidate.feasible_under_tau:
                continue
            if _dominates(feasible, candidate) and candidate.surrogate_score > feasible.surrogate_score + 1e-9:
                dominated_infeasible_misranks.append(
                    {"feasible_row": feasible.row_id, "infeasible_row": candidate.row_id}
                )

    feasible_rows = [row for row in rows if row.feasible_under_tau]
    feasible_reliability_components = [
        row.reliability_violation_component for row in feasible_rows
    ]

    empty_rows = [
        row
        for row in rows
        if row.selected_edge_count == 0 or "empty" in row.policy_label.lower()
    ]
    reward_values = sorted(row.surrogate_score for row in rows)
    top_quartile_floor = reward_values[int(0.75 * (len(reward_values) - 1))] if reward_values else 0.0
    high_reward_empty = [
        row.row_id for row in empty_rows if row.surrogate_score >= top_quartile_floor
    ]

    full_rows = [
        row
        for row in rows
        if (
            row.candidate_edge_count > 0
            and row.selected_edge_count >= row.candidate_edge_count
        )
        or "full" in row.policy_label.lower()
    ]
    sparse_feasible_rows = [
        row
        for row in rows
        if row.feasible_under_tau
        and row.candidate_edge_count > 0
        and row.selected_edge_count < row.candidate_edge_count
    ]
    full_dominance_issues = []
    for full in full_rows:
        for sparse in sparse_feasible_rows:
            if full.scenario_id != sparse.scenario_id:
                continue
            if (
                sparse.latency <= full.latency
                and sparse.energy <= full.energy
                and full.surrogate_score > sparse.surrogate_score + 1e-9
            ):
                full_dominance_issues.append(
                    {"full_row": full.row_id, "sparse_feasible_row": sparse.row_id}
                )

    objective_inversions = []
    objective_comparisons = 0
    for left in rows:
        for right in rows:
            if left.row_id == right.row_id:
                continue
            if left.scenario_id != right.scenario_id:
                continue
            if _objective_key(left) < _objective_key(right) and left.surrogate_score < right.surrogate_score - 1e-9:
                objective_comparisons += 1
                objective_inversions.append(
                    {"better_objective_row": left.row_id, "higher_reward_row": right.row_id}
                )
            elif _objective_key(left) < _objective_key(right):
                objective_comparisons += 1
    inversion_rate = (
        len(objective_inversions) / objective_comparisons
        if objective_comparisons
        else 0.0
    )
    return {
        "feasible_not_ranked_below_dominated_infeasible": {
            "passed": not dominated_infeasible_misranks,
            "issues": dominated_infeasible_misranks,
        },
        "above_tau_reliability_component_zero": {
            "passed": all(abs(value) <= 1e-12 for value in feasible_reliability_components),
            "max_feasible_reliability_component": max(feasible_reliability_components, default=0.0),
            "tau_requirement_min": tau_requirement_min,
        },
        "empty_topology_not_high_reward": {
            "passed": not high_reward_empty,
            "high_reward_empty_rows": high_reward_empty,
        },
        "full_topology_not_automatic_sparse_dominator": {
            "passed": not full_dominance_issues,
            "issues": full_dominance_issues,
        },
        "reward_rank_broadly_matches_objective_order": {
            "passed": inversion_rate <= 0.25,
            "inversion_count": len(objective_inversions),
            "comparison_count": objective_comparisons,
            "inversion_rate": inversion_rate,
            "sample_inversions": objective_inversions[:20],
        },
    }


def _dominated_flags(rows: list[dict[str, object]]) -> list[bool]:
    flags: list[bool] = []
    for index, candidate in enumerate(rows):
        dominated = False
        for other_index, other in enumerate(rows):
            if index == other_index:
                continue
            if str(other["scenario_id"]) != str(candidate["scenario_id"]):
                continue
            if _dominates_mapping(other, candidate):
                dominated = True
                break
        flags.append(dominated)
    return flags


def _dominates(left: ObjectiveSurfaceRecord, right: ObjectiveSurfaceRecord) -> bool:
    return (
        left.consensus_success_probability >= right.consensus_success_probability
        and left.latency <= right.latency
        and left.energy <= right.energy
        and (
            left.consensus_success_probability > right.consensus_success_probability
            or left.latency < right.latency
            or left.energy < right.energy
        )
    )


def _dominates_mapping(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return (
        float(left["consensus_success_probability"]) >= float(right["consensus_success_probability"])
        and float(left["latency"]) <= float(right["latency"])
        and float(left["energy"]) <= float(right["energy"])
        and (
            float(left["consensus_success_probability"]) > float(right["consensus_success_probability"])
            or float(left["latency"]) < float(right["latency"])
            or float(left["energy"]) < float(right["energy"])
        )
    )


def _objective_key(row: ObjectiveSurfaceRecord) -> tuple[int, float, float, int, str]:
    return (
        0 if row.feasible_under_tau else 1,
        row.latency,
        row.energy,
        row.selected_edge_count,
        row.row_id,
    )


def _finite_float(record: Mapping[str, object], key: str) -> float:
    value = float(record[key])
    if not isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value
