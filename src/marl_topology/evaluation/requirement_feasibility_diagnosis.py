"""Requirement-anchored feasibility diagnosis for Stage 5.0h."""

from __future__ import annotations

from collections import Counter
from math import isfinite
from typing import Iterable, Mapping

from marl_topology.metrics import require_registered_metrics

from .calibration_fixture_suite import (
    STAGE5_0F_SOURCE_SCOPE,
    build_stage5_0f_tau_calibration_fixture_suite_report,
)


STAGE5_0H_REQUIREMENT_DIAGNOSIS_STAGE_ID = (
    "stage_5_0h_requirement_anchored_feasibility_diagnosis"
)
TAU_REQUIREMENT_MIN = 0.9
TAU_STRESS_CANDIDATES = (0.95, 0.99)
TAU_DIAGNOSTIC_VALUES = (0.5, 0.75)
SUPPORTED_FAILURE_REASONS = (
    "link_budget_failure",
    "deadline_failure",
    "retransmission_insufficient",
    "topology_candidate_failure",
    "interference_failure",
    "pbft_quorum_failure",
    "primary_distribution_failure",
    "resource_budget_failure",
    "modeling_suspicious",
    "unknown",
)
PARAMETER_SANITY_STATUSES = (
    "reasonable",
    "too_strict",
    "too_loose",
    "unknown_needs_reference",
)
_REGISTERED_REPORT_METRICS = (
    "consensus_success_probability",
    "latency",
    "energy",
    "topology_diagnostics",
)
_DOMINANT_REASON_PRIORITY = (
    "primary_distribution_failure",
    "pbft_quorum_failure",
    "interference_failure",
    "deadline_failure",
    "retransmission_insufficient",
    "resource_budget_failure",
    "link_budget_failure",
    "topology_candidate_failure",
    "modeling_suspicious",
    "unknown",
)


def build_stage5_0h_requirement_feasibility_diagnosis(
    *,
    tau_requirement_min: float = TAU_REQUIREMENT_MIN,
    tau_stress_candidates: Iterable[float] = TAU_STRESS_CANDIDATES,
    tau_diagnostic_values: Iterable[float] = TAU_DIAGNOSTIC_VALUES,
) -> dict[str, object]:
    """Build a diagnosis report anchored to the owner reliability requirement."""

    _validate_tau_requirement(tau_requirement_min)
    stress_values = tuple(_validated_tau(value) for value in tau_stress_candidates)
    diagnostic_values = tuple(_validated_tau(value) for value in tau_diagnostic_values)
    if any(value < tau_requirement_min for value in stress_values):
        raise ValueError("tau_stress_candidates must be >= tau_requirement_min")
    if any(value >= tau_requirement_min for value in diagnostic_values):
        raise ValueError("tau_diagnostic_values must be lower than tau_requirement_min")

    require_registered_metrics(_REGISTERED_REPORT_METRICS)
    source_report = build_stage5_0f_tau_calibration_fixture_suite_report()
    source_rows = [dict(_mapping(row)) for row in _sequence(source_report["topology_evaluation_rows"])]
    row_diagnosis = [_diagnose_row(row, tau_requirement_min) for row in source_rows]
    family_summary = _family_summary(row_diagnosis)

    return {
        "stage": STAGE5_0H_REQUIREMENT_DIAGNOSIS_STAGE_ID,
        "source_stage": source_report["stage"],
        "source_scope": STAGE5_0F_SOURCE_SCOPE,
        "source_kind": "stage5_0f_alpha_fixture_suite",
        "tau_requirement_min": tau_requirement_min,
        "tau_stress_candidates": list(stress_values),
        "tau_diagnostic_values": list(diagnostic_values),
        "tau_candidate_semantics": (
            "candidate language is diagnostic only; tau_requirement_min is the "
            "requirement baseline and must not be lowered automatically"
        ),
        "objective_feasibility_condition": (
            "consensus_success_probability >= tau_requirement_min"
        ),
        "row_diagnosis": row_diagnosis,
        "family_summary": family_summary,
        "topology_failure_summary": _topology_failure_summary(row_diagnosis),
        "parameter_sanity_table": _parameter_sanity_table(),
        "feasibility_envelope_plan": _feasibility_envelope_plan(),
        "stress_summary": _threshold_summary(row_diagnosis, stress_values),
        "diagnostic_value_summary": _threshold_summary(row_diagnosis, diagnostic_values),
        "supported_failure_reasons": list(SUPPORTED_FAILURE_REASONS),
        "checks": _checks(row_diagnosis, family_summary, tau_requirement_min),
    }


def _diagnose_row(
    row: Mapping[str, object],
    tau_requirement_min: float,
) -> dict[str, object]:
    probability = float(row["consensus_success_probability"])
    requirement_met = probability >= tau_requirement_min
    reason = None if requirement_met else _classify_failure_reason(row)
    return {
        "scenario_family": row["scenario_family"],
        "scenario_id": row["scenario_id"],
        "fixture_id": row["fixture_id"],
        "topology_name": row["topology_name"],
        "topology_family": row["topology_family"],
        "selected_edge_count": row["selected_edge_count"],
        "is_full_graph_baseline": row["is_full_graph_baseline"],
        "is_oracle_candidate": row["is_oracle_candidate"],
        "is_deployment_actor_input": row["is_deployment_actor_input"],
        "consensus_success_probability": probability,
        "latency": row["latency"],
        "energy": row["energy"],
        "per_primary_reliability": row["per_primary_reliability"],
        "diagnostic_flags": row["diagnostic_flags"],
        "tau_requirement_min": tau_requirement_min,
        "requirement_met": requirement_met,
        "failure_reason": reason,
        "diagnostic_notes": _diagnostic_notes(row, reason),
    }


def _classify_failure_reason(row: Mapping[str, object]) -> str:
    flags = set(_sequence(row["diagnostic_flags"]))
    topology_family = str(row["topology_family"])
    topology_name = str(row["topology_name"])
    selected_edge_count = int(row["selected_edge_count"])
    is_full_graph = bool(row["is_full_graph_baseline"])

    if "weak_primary_distribution" in flags:
        if "balanced_dense_baseline" in flags:
            return "pbft_quorum_failure"
        return "primary_distribution_failure"
    if "same_resource_interference" in flags and is_full_graph:
        return "interference_failure"
    if "deadline_tight_retransmission" in flags:
        if topology_family == "sparse_candidate":
            return "retransmission_insufficient"
        return "deadline_failure"
    if "required_time_capped" in flags:
        if selected_edge_count == 0:
            return "topology_candidate_failure"
        return "resource_budget_failure"
    if "nlos_penalty_visible" in flags or "non_saturated_boundary" in flags:
        return "link_budget_failure"
    if selected_edge_count == 0 or topology_family == "weak_or_disconnected_baseline":
        return "topology_candidate_failure"
    if "dense_full_graph_baseline" in topology_name and is_full_graph:
        return "resource_budget_failure"
    return "unknown"


def _diagnostic_notes(row: Mapping[str, object], reason: str | None) -> str:
    if reason is None:
        return "meets tau_requirement_min without changing requirement baseline"
    if reason == "link_budget_failure":
        return "communication reliability is below the requirement; audit path loss, LoS/NLoS, bandwidth, power, and distance before lowering tau"
    if reason == "deadline_failure":
        return "scheduled communication exists but deadline budget is too tight for the requirement"
    if reason == "retransmission_insufficient":
        return "candidate improves reliability but available retry budget does not reach tau_requirement_min"
    if reason == "topology_candidate_failure":
        return "selected topology is weak, disconnected, or too sparse for the requirement"
    if reason == "interference_failure":
        return "dense/full topology is penalized by shared-resource interference or resource contention"
    if reason == "pbft_quorum_failure":
        return "PBFT three-phase quorum amplification keeps consensus below the requirement despite nonzero communication probability"
    if reason == "primary_distribution_failure":
        return "uniform initiator distribution exposes weak primary or edge-primary reliability spread"
    if reason == "resource_budget_failure":
        return "finite transmission/resource budget appears capped before the requirement can be reached"
    if reason == "modeling_suspicious":
        return "row should be reviewed because its declared alpha behavior may not match Stage 3 physics"
    return "insufficient diagnostic signal; add a more specific Stage 3/4 sensor"


def _family_summary(row_diagnosis: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for family_id in sorted({str(row["scenario_family"]) for row in row_diagnosis}):
        rows = [row for row in row_diagnosis if row["scenario_family"] == family_id]
        feasible = [row for row in rows if bool(row["requirement_met"])]
        infeasible = [row for row in rows if not bool(row["requirement_met"])]
        reason_counts = Counter(str(row["failure_reason"]) for row in infeasible)
        dominant_reason = _dominant_reason(reason_counts)
        result.append(
            {
                "scenario_family": family_id,
                "topology_count": len(rows),
                "feasible_topology_count": len(feasible),
                "infeasible_topology_count": len(infeasible),
                "feasible_topologies": [str(row["topology_name"]) for row in feasible],
                "infeasible_topologies": [str(row["topology_name"]) for row in infeasible],
                "dominant_failure_reason": dominant_reason,
                "failure_reason_counts": dict(sorted(reason_counts.items())),
                "diagnostic_notes": _family_notes(family_id, dominant_reason),
            }
        )
    return result


def _topology_failure_summary(
    row_diagnosis: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "topology_name": row["topology_name"],
            "topology_family": row["topology_family"],
            "scenario_family": row["scenario_family"],
            "requirement_met": row["requirement_met"],
            "failure_reason": row["failure_reason"],
            "diagnostic_notes": row["diagnostic_notes"],
        }
        for row in row_diagnosis
    ]


def _dominant_reason(reason_counts: Counter[str]) -> str | None:
    if not reason_counts:
        return None
    best_count = max(reason_counts.values())
    tied = {reason for reason, count in reason_counts.items() if count == best_count}
    for reason in _DOMINANT_REASON_PRIORITY:
        if reason in tied:
            return reason
    return sorted(tied)[0]


def _family_notes(family_id: str, dominant_reason: str | None) -> str:
    if dominant_reason is None:
        return "at least one topology meets tau_requirement_min; compare latency and energy before preferring dense baselines"
    if family_id == "weak_primary_distribution":
        return "inspect per-primary reliability spread and PBFT quorum amplification before changing tau"
    if family_id == "same_resource_interference":
        return "resource orthogonalization should be tested before treating dense topology as infeasible in principle"
    if family_id == "deadline_tight_retransmission":
        return "deadline and retransmission budget are the next control knobs to audit"
    if family_id == "blocked_or_nlos_urban":
        return "geometry and link budget are the next control knobs to audit"
    return "diagnose environment/protocol controls before lowering tau_requirement_min"


def _parameter_sanity_table() -> list[dict[str, object]]:
    return [
        _parameter("tx_power", "not exposed in Stage 5.0f alpha rows", "unknown_needs_reference", "audit Stage 3.6 link records before changing"),
        _parameter("bandwidth", "not exposed in Stage 5.0f alpha rows", "unknown_needs_reference", "needed for finite-blocklength feasibility envelope"),
        _parameter("noise", "not exposed in Stage 5.0f alpha rows", "unknown_needs_reference", "needed to separate SINR failure from topology failure"),
        _parameter("carrier_frequency", "not exposed in Stage 5.0f alpha rows", "unknown_needs_reference", "needed for path-loss realism"),
        _parameter("path_loss_los_nlos_penalty", "represented only by family diagnostic flags", "unknown_needs_reference", "blocked/NLoS family suggests sensitivity but not calibrated realism"),
        _parameter("interference_model", "same_resource_interference family uses declared alpha probabilities", "unknown_needs_reference", "requires Stage 3 resource-group audit"),
        _parameter("payload_bits", "not exposed in Stage 5.0f alpha rows", "unknown_needs_reference", "needed for finite-blocklength and deadline diagnosis"),
        _parameter("deadline", "deadline_tight_retransmission is a stress fixture", "too_strict", "stress row intentionally diagnoses deadline pressure"),
        _parameter("attempt_duration", "not exposed in Stage 5.0f alpha rows", "unknown_needs_reference", "needed to diagnose retransmission feasibility"),
        _parameter("max_retransmissions", "not exposed in Stage 5.0f alpha rows", "unknown_needs_reference", "needed to separate retry-budget failure from link-budget failure"),
        _parameter("rsu_vehicle_distances", "not exposed in Stage 5.0f alpha rows", "unknown_needs_reference", "needed for geometry-backed calibration"),
        _parameter("candidate_edge_radius", "not exposed in Stage 5.0f alpha rows", "unknown_needs_reference", "needed to diagnose topology candidate expansion"),
        _parameter("pbft_n_f_quorum", "n=4 f=1 quorum=3 minimal PBFT fixture", "reasonable", "valid minimal committee, but larger n should be swept later"),
        _parameter("fault_filter_mode", "none in Stage 5.0f alpha suite", "too_loose", "optimistic relative to conservative remove_largest comparison"),
    ]


def _parameter(
    name: str,
    representation: str,
    status: str,
    note: str,
) -> dict[str, object]:
    if status not in PARAMETER_SANITY_STATUSES:
        raise ValueError(f"unsupported parameter sanity status: {status}")
    return {
        "parameter": name,
        "current_representation": representation,
        "sanity_status": status,
        "diagnostic_note": note,
    }


def _feasibility_envelope_plan() -> list[dict[str, object]]:
    return [
        _sweep("bandwidth_sweep", "bandwidth_hz", "separate finite-blocklength capacity from topology failure", "higher bandwidth should improve link reliability or reduce required duration"),
        _sweep("tx_power_sweep", "tx_power_w", "separate link-budget failure from topology candidate failure", "higher tx power should improve SINR-limited rows"),
        _sweep("deadline_sweep", "deadline_s", "separate deadline failure from intrinsic link failure", "longer deadline should improve retransmission-limited rows"),
        _sweep("payload_sweep", "payload_bits", "separate message-size pressure from topology failure", "larger payload should reduce feasible envelope"),
        _sweep("rsu_height_placement_sweep", "rsu_height_and_position", "separate NLoS geometry from channel budget", "higher or better-placed RSUs should recover some LoS rows"),
        _sweep("resource_orthogonalization_sweep", "channel_resource_assignment", "separate full-graph interference from dense-topology infeasibility", "orthogonal resources should improve dense/full rows"),
        _sweep("fault_filter_mode_comparison", "fault_filter_mode", "compare optimistic none with conservative remove_largest", "remove_largest should lower or preserve reliability"),
        _sweep("topology_candidate_expansion", "candidate_edge_radius_and_candidate_edges", "separate missing candidates from environment infeasibility", "expanded candidate sets should improve topology-candidate failures if resources suffice"),
    ]


def _sweep(
    sweep_id: str,
    controlled_parameter: str,
    reason: str,
    expected_signal: str,
) -> dict[str, object]:
    return {
        "sweep_id": sweep_id,
        "controlled_parameter": controlled_parameter,
        "reason": reason,
        "expected_signal": expected_signal,
        "blocked_changes": [
            "do not lower tau_requirement_min during sweep",
            "do not implement reward",
            "do not train models",
            "do not migrate v5 code",
        ],
    }


def _threshold_summary(
    row_diagnosis: list[dict[str, object]],
    thresholds: tuple[float, ...],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for tau in thresholds:
        feasible_rows = [
            row
            for row in row_diagnosis
            if float(row["consensus_success_probability"]) >= tau
        ]
        summaries.append(
            {
                "tau_value": tau,
                "feasible_topology_count": len(feasible_rows),
                "infeasible_topology_count": len(row_diagnosis) - len(feasible_rows),
                "diagnostic_only": tau < TAU_REQUIREMENT_MIN,
                "stress_only": tau >= TAU_REQUIREMENT_MIN,
            }
        )
    return summaries


def _checks(
    row_diagnosis: list[dict[str, object]],
    family_summary: list[dict[str, object]],
    tau_requirement_min: float,
) -> dict[str, object]:
    infeasible = [row for row in row_diagnosis if not bool(row["requirement_met"])]
    feasible = [row for row in row_diagnosis if bool(row["requirement_met"])]
    return {
        "tau_requirement_min_recorded": tau_requirement_min == TAU_REQUIREMENT_MIN,
        "objective_direction_is_greater_equal": True,
        "final_tau_below_requirement_selected": False,
        "final_tau_selected": False,
        "all_infeasible_rows_have_failure_reason": all(
            row["failure_reason"] in SUPPORTED_FAILURE_REASONS for row in infeasible
        ),
        "family_dominant_failure_reasons_present": all(
            row["dominant_failure_reason"] in SUPPORTED_FAILURE_REASONS or row["dominant_failure_reason"] is None
            for row in family_summary
        ),
        "feasible_family_count": len(
            {row["scenario_family"] for row in feasible}
        ),
        "infeasible_family_count": len(
            {
                row["scenario_family"]
                for row in family_summary
                if int(row["feasible_topology_count"]) == 0
            }
        ),
        "full_graph_not_oracle": all(
            not bool(row["is_oracle_candidate"])
            for row in row_diagnosis
            if bool(row["is_full_graph_baseline"])
        ),
        "oracle_labels_not_actor_inputs": all(
            not bool(row["is_deployment_actor_input"]) for row in row_diagnosis
        ),
        "simulation_parameters_changed_to_force_feasibility": False,
        "reward_implemented": False,
        "training_run": False,
        "v5_code_migrated": False,
    }


def _validate_tau_requirement(value: float) -> None:
    _validated_tau(value)
    if value < TAU_REQUIREMENT_MIN:
        raise ValueError("tau_requirement_min must not be below 0.9")


def _validated_tau(value: float) -> float:
    if not isfinite(value):
        raise ValueError("tau must be finite")
    if not 0.0 <= value <= 1.0:
        raise ValueError("tau must be in [0, 1]")
    return float(value)


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("expected mapping payload")
    return value


def _sequence(value: object) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("expected sequence payload")
    return tuple(value)
