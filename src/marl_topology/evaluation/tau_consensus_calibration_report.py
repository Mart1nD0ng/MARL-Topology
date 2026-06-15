"""Stage 5.0d tau-consensus calibration report sensor.

The report is executable, but it is still decision support only: it requires
owner-declared candidate tau values and never selects the final threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Mapping

from marl_topology.metrics import REGISTERED_METRICS, require_registered_metrics

from .calibration_fixture_suite import STAGE5_0F_TAU_FIXTURE_SUITE_STAGE_ID
from .stage4_boundary_audit import (
    STAGE4_8_BOUNDARY_AUDIT_STAGE_ID,
    build_stage4_8_boundary_audit_report,
)


STAGE5_0D_TAU_CALIBRATION_REPORT_STAGE_ID = (
    "stage_5_0d_tau_consensus_calibration_report_implementation_without_tau_selection"
)
STAGE5_0D_SOURCE_SCOPE = "stage4_8_smoke_test_only_not_calibration_set"
STAGE5_0D_REPORT_ID = "stage5_0d_tau_consensus_stage4_8_smoke_report"
SOURCE_KIND_STAGE4_8 = "stage4_8_smoke"
SOURCE_KIND_STAGE5_0F = "stage5_0f_alpha_fixture_suite"
REQUIRED_REPORT_METRICS = (
    "consensus_success_probability",
    "latency",
    "energy",
    "topology_diagnostics",
)


@dataclass(frozen=True, slots=True)
class TauCandidate:
    """Owner-declared candidate threshold row for the report."""

    tau_candidate: float
    tau_source: str = "owner_declared"
    tau_owner_note: str = ""
    is_default: bool = False
    is_selected: bool = False

    def __post_init__(self) -> None:
        if not isfinite(self.tau_candidate):
            raise ValueError("tau_candidate must be finite")
        if not 0.0 <= self.tau_candidate <= 1.0:
            raise ValueError("tau_candidate must be in [0, 1]")
        if not self.tau_source.strip():
            raise ValueError("tau_source must be non-empty")
        if self.is_default:
            raise ValueError("Stage 5.0d does not allow default tau candidates")
        if self.is_selected:
            raise ValueError("Stage 5.0d does not select final tau_consensus")

    def to_payload(self) -> dict[str, object]:
        return {
            "tau_candidate": self.tau_candidate,
            "tau_source": self.tau_source,
            "tau_owner_note": self.tau_owner_note,
            "is_default": self.is_default,
            "is_selected": self.is_selected,
        }


@dataclass(frozen=True, slots=True)
class TauCalibrationReportConfig:
    """Configuration for a report-only tau calibration view."""

    tau_candidates: tuple[TauCandidate, ...]
    scenario_set_id: str = "stage5_0d_stage4_8_boundary_smoke"
    scenario_family: str = "stage4_8_boundary_smoke"
    owner_scope_note: str = (
        "Stage 5.0d uses Stage 4.8 boundary rows as smoke-test input only; "
        "it is not a final calibration set."
    )

    def __post_init__(self) -> None:
        if not self.tau_candidates:
            raise ValueError("at least one owner-declared tau candidate is required")
        values = [candidate.tau_candidate for candidate in self.tau_candidates]
        if len(values) != len(set(values)):
            raise ValueError("tau_candidate values must be unique")
        if not self.scenario_set_id.strip():
            raise ValueError("scenario_set_id must be non-empty")
        if not self.scenario_family.strip():
            raise ValueError("scenario_family must be non-empty")


def build_tau_consensus_calibration_report(
    tau_candidates: Iterable[float | TauCandidate],
    *,
    tau_source: str = "owner_declared",
    tau_owner_note: str = "",
    source_report: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a report-only tau-consensus feasibility view.

    Candidate tau values must be supplied by the caller. The function computes
    feasibility views for those candidates, but leaves final selection to the
    owner decision packet.
    """

    config = TauCalibrationReportConfig(
        tau_candidates=_candidate_tuple(
            tau_candidates,
            tau_source=tau_source,
            tau_owner_note=tau_owner_note,
        ),
    )
    if source_report is None:
        source_report = build_stage4_8_boundary_audit_report()
    source_kind = _validate_source_report(source_report)
    require_registered_metrics(REQUIRED_REPORT_METRICS)

    scenario_manifest = _scenario_manifest(config, source_report, source_kind)
    topology_rows = _topology_evaluation_rows(config, source_report, source_kind)
    candidate_tau_rows = [candidate.to_payload() for candidate in config.tau_candidates]
    protocol_rows = _protocol_regime_rows(source_report, source_kind)
    scenario_summary = _scenario_summary_rows(config, scenario_manifest, topology_rows)
    tau_feasibility_summary = [
        _tau_feasibility_summary_row(candidate, scenario_family, family_rows, source_kind)
        for candidate in config.tau_candidates
        for scenario_family, family_rows in _rows_by_scenario_family(topology_rows).items()
    ]
    topology_by_tau_detail = [
        _topology_by_tau_detail_row(candidate, row)
        for candidate in config.tau_candidates
        for row in topology_rows
    ]

    return {
        "stage": STAGE5_0D_TAU_CALIBRATION_REPORT_STAGE_ID,
        "report_id": STAGE5_0D_REPORT_ID,
        "source_stage": source_report["stage"],
        "source_scope": _source_scope(source_report, source_kind),
        "source_kind": source_kind,
        "calibration_ready": source_kind == SOURCE_KIND_STAGE5_0F,
        "tau_selected": False,
        "final_tau_consensus": None,
        "scenario_manifest": scenario_manifest,
        "topology_evaluation_rows": topology_rows,
        "candidate_tau_rows": candidate_tau_rows,
        "protocol_regime_rows": protocol_rows,
        "scenario_summary": scenario_summary,
        "tau_feasibility_summary": tau_feasibility_summary,
        "topology_by_tau_detail": topology_by_tau_detail,
        "owner_decision_packet": _owner_decision_packet(config, scenario_manifest, source_kind),
        "metric_governance": {
            "registered_metric_names": list(REGISTERED_METRICS.keys()),
            "metric_valued_fields": list(REQUIRED_REPORT_METRICS),
            "metric_valued_fields_registered": True,
            "new_metric_names_introduced": [],
        },
        "checks": _checks(config, topology_rows, source_kind),
    }


def _candidate_tuple(
    tau_candidates: Iterable[float | TauCandidate],
    *,
    tau_source: str,
    tau_owner_note: str,
) -> tuple[TauCandidate, ...]:
    candidates: list[TauCandidate] = []
    for value in tau_candidates:
        if isinstance(value, TauCandidate):
            candidates.append(value)
        else:
            candidates.append(
                TauCandidate(
                    tau_candidate=float(value),
                    tau_source=tau_source,
                    tau_owner_note=tau_owner_note,
                )
            )
    return tuple(sorted(candidates, key=lambda candidate: candidate.tau_candidate))


def _validate_source_report(source_report: Mapping[str, object]) -> str:
    stage = source_report.get("stage")
    if stage == STAGE4_8_BOUNDARY_AUDIT_STAGE_ID:
        if not source_report.get("audit_rows"):
            raise ValueError("Stage 4.8 source report must contain audit_rows")
        return SOURCE_KIND_STAGE4_8
    if stage == STAGE5_0F_TAU_FIXTURE_SUITE_STAGE_ID:
        if not source_report.get("topology_evaluation_rows"):
            raise ValueError("Stage 5.0f source report must contain topology_evaluation_rows")
        if not source_report.get("scenario_manifest"):
            raise ValueError("Stage 5.0f source report must contain scenario_manifest")
        return SOURCE_KIND_STAGE5_0F
    raise ValueError("unsupported tau calibration source report stage")


def _scenario_manifest(
    config: TauCalibrationReportConfig,
    source_report: Mapping[str, object],
    source_kind: str,
) -> list[dict[str, object]]:
    if source_kind == SOURCE_KIND_STAGE5_0F:
        return [dict(_mapping(row)) for row in _sequence(source_report["scenario_manifest"])]
    scenario = _mapping(source_report["scenario"])
    return [
        {
            "scenario_set_id": config.scenario_set_id,
            "scenario_family": config.scenario_family,
            "scenario_id": scenario["scenario_id"],
            "fixture_id": "stage4_8_boundary_audit_rows",
            "physics_regime": scenario["physics_regime"],
            "channel_regime": "stage3_6_urlcc_finite_blocklength_v1",
            "link_transmission_regime": "urlcc_finite_blocklength_v1",
            "network_regime": "stage3_network_communication_v1",
            "pbft_model_id": "pbft_expected_initiator_mean_field_v1",
            "accounting_model_id": "stage4_protocol_scheduled_phase_occupancy_v1",
            "deterministic_seed_policy": "deterministic_stage4_8_boundary_rows",
            "owner_scope_note": config.owner_scope_note,
        }
    ]


def _topology_evaluation_rows(
    config: TauCalibrationReportConfig,
    source_report: Mapping[str, object],
    source_kind: str,
) -> list[dict[str, object]]:
    if source_kind == SOURCE_KIND_STAGE5_0F:
        return _stage5_0f_topology_rows(source_report)
    scenario = _mapping(source_report["scenario"])
    rows = []
    for row in _sequence(source_report["audit_rows"]):
        row_mapping = _mapping(row)
        diagnostic_flags = tuple(row_mapping.get("diagnostic_flags", ()))
        topology_name = str(row_mapping["topology_name"])
        rows.append(
            {
                "scenario_set_id": config.scenario_set_id,
                "scenario_family": config.scenario_family,
                "scenario_id": scenario["scenario_id"],
                "fixture_id": "stage4_8_boundary_audit_rows",
                "topology_name": topology_name,
                "topology_family": _topology_family(row_mapping, diagnostic_flags),
                "selected_edge_count": row_mapping["selected_edge_count"],
                "is_full_graph_baseline": bool(row_mapping["is_full_graph_baseline"]),
                "is_oracle_candidate": bool(row_mapping["is_oracle_candidate"]),
                "is_deployment_actor_input": bool(row_mapping["is_deployment_actor_input"]),
                "consensus_success_probability": row_mapping["consensus_success_probability"],
                "latency": row_mapping["scheduled_latency_s"],
                "successful_delivery_latency": row_mapping["successful_delivery_latency_s"],
                "energy": row_mapping["energy_j"],
                "link_deadline_delivery_probability": row_mapping[
                    "link_deadline_delivery_probability"
                ],
                "network_deadline_delivery_probability": row_mapping[
                    "network_deadline_delivery_probability"
                ],
                "per_primary_reliability": row_mapping["per_primary_reliability"],
                "diagnostic_flags": list(diagnostic_flags),
            }
        )
    return rows


def _stage5_0f_topology_rows(source_report: Mapping[str, object]) -> list[dict[str, object]]:
    required = {
        "scenario_set_id",
        "scenario_family",
        "scenario_id",
        "fixture_id",
        "topology_name",
        "topology_family",
        "selected_edge_count",
        "is_full_graph_baseline",
        "is_oracle_candidate",
        "is_deployment_actor_input",
        "consensus_success_probability",
        "latency",
        "energy",
        "per_primary_reliability",
        "diagnostic_flags",
    }
    rows: list[dict[str, object]] = []
    for raw_row in _sequence(source_report["topology_evaluation_rows"]):
        row = dict(_mapping(raw_row))
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Stage 5.0f topology row missing fields: {missing}")
        if not 0.0 <= float(row["consensus_success_probability"]) <= 1.0:
            raise ValueError("consensus_success_probability must be in [0, 1]")
        if float(row["latency"]) < 0.0 or float(row["energy"]) < 0.0:
            raise ValueError("latency and energy must be nonnegative")
        if row["is_full_graph_baseline"] and row["is_oracle_candidate"]:
            raise ValueError("full graph baseline must not be oracle")
        if row["is_deployment_actor_input"]:
            raise ValueError("deployment actor input labels are forbidden")
        rows.append(row)
    return rows


def _protocol_regime_rows(
    source_report: Mapping[str, object],
    source_kind: str,
) -> list[dict[str, object]]:
    if source_kind == SOURCE_KIND_STAGE5_0F:
        return [dict(_mapping(row)) for row in _sequence(source_report["protocol_regime_rows"])]
    scenario = _mapping(source_report["scenario"])
    return [
        {
            "node_count": len(tuple(scenario["node_ids"])),
            "fault_tolerance": 1,
            "phase_budgets_s": "not_exposed_by_stage4_8_smoke_rows",
            "primary_distribution": "uniform",
            "fault_filter_mode": "remove_largest_for_link-derived_rows_or_none_for_synthetic_rows",
            "view_change_mode": "deferred",
            "mean_field_assumption": True,
            "urlcc_finite_blocklength_v1": True,
            "pbft_expected_initiator_mean_field_v1": True,
        }
    ]


def _scenario_summary_rows(
    config: TauCalibrationReportConfig,
    scenario_manifest: list[dict[str, object]],
    topology_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    manifest_by_family = _rows_by_scenario_family(scenario_manifest)
    topology_by_family = _rows_by_scenario_family(topology_rows)
    summaries: list[dict[str, object]] = []
    for scenario_family, family_rows in topology_by_family.items():
        family_manifest = manifest_by_family.get(scenario_family, [])
        topology_families = {str(row["topology_family"]) for row in family_rows}
        summaries.append(
            {
                "scenario_family": scenario_family,
                "scenario_count": len({row["scenario_id"] for row in family_manifest}),
                "topology_count": len(family_rows),
                "candidate_tau_count": len(config.tau_candidates),
                "physics_regime_set": sorted(
                    {str(row.get("physics_regime", "unknown")) for row in family_manifest}
                ),
                "protocol_model_set": sorted(
                    {str(row.get("pbft_model_id", "unknown")) for row in family_manifest}
                ),
                "has_failed_scheduled_message_case": any(
                    row["topology_name"] == "failed_scheduled_message"
                    or "failed_scheduled_message" in row.get("diagnostic_flags", ())
                    or "unreachable_reliability_target" in row.get("diagnostic_flags", ())
                    for row in family_rows
                ),
                "has_full_graph_baseline": any(
                    bool(row["is_full_graph_baseline"]) for row in family_rows
                ),
                "has_sparse_candidate": "sparse_candidate" in topology_families,
                "has_oracle_candidate_diagnostic": any(
                    bool(row["is_oracle_candidate"]) for row in family_rows
                ),
            }
        )
    return summaries


def _tau_feasibility_summary_row(
    candidate: TauCandidate,
    scenario_family: str,
    topology_rows: list[dict[str, object]],
    source_kind: str,
) -> dict[str, object]:
    tau = candidate.tau_candidate
    feasible = [
        row
        for row in topology_rows
        if float(row["consensus_success_probability"]) >= tau
    ]
    full_feasible = [
        row for row in feasible if bool(row["is_full_graph_baseline"])
    ]
    non_full_feasible = [
        row for row in feasible if not bool(row["is_full_graph_baseline"])
    ]
    source_flag = (
        "stage5_0f_alpha_fixture_suite"
        if source_kind == SOURCE_KIND_STAGE5_0F
        else "stage4_8_smoke_only"
    )
    flags = [source_flag, "no_default_tau", "no_final_tau_selected"]
    if not feasible:
        flags.append("no_feasible_topology_at_tau")
    if full_feasible and len(feasible) == len(full_feasible):
        flags.append("full_graph_only_feasible")
    return {
        "tau_candidate": tau,
        "scenario_family": scenario_family,
        "feasible_topology_count": len(feasible),
        "infeasible_topology_count": len(topology_rows) - len(feasible),
        "feasible_non_full_topology_count": len(non_full_feasible),
        "full_graph_feasible": bool(full_feasible),
        "full_graph_only_feasible": bool(full_feasible and len(feasible) == len(full_feasible)),
        "oracle_candidate_feasible": any(bool(row["is_oracle_candidate"]) for row in feasible),
        "lowest_feasible_latency": _minimum_or_none(feasible, "latency"),
        "lowest_feasible_energy": _minimum_or_none(feasible, "energy"),
        "pareto_feasible_count": _pareto_count(feasible),
        "weak_primary_case_count": sum(1 for row in topology_rows if _is_weak_primary_case(row)),
        "diagnostic_flags": flags,
    }


def _topology_by_tau_detail_row(
    candidate: TauCandidate,
    row: Mapping[str, object],
) -> dict[str, object]:
    tau = candidate.tau_candidate
    return {
        "tau_candidate": tau,
        "scenario_id": row["scenario_id"],
        "scenario_family": row["scenario_family"],
        "fixture_id": row["fixture_id"],
        "topology_name": row["topology_name"],
        "topology_family": row["topology_family"],
        "selected_edge_count": row["selected_edge_count"],
        "consensus_success_probability": row["consensus_success_probability"],
        "latency": row["latency"],
        "energy": row["energy"],
        "reliability_feasible": float(row["consensus_success_probability"]) >= tau,
        "is_full_graph_baseline": row["is_full_graph_baseline"],
        "is_oracle_candidate": row["is_oracle_candidate"],
        "is_deployment_actor_input": row["is_deployment_actor_input"],
        "per_primary_reliability": row["per_primary_reliability"],
        "diagnostic_flags": row["diagnostic_flags"],
    }


def _owner_decision_packet(
    config: TauCalibrationReportConfig,
    scenario_manifest: list[dict[str, object]],
    source_kind: str,
) -> dict[str, object]:
    scenario_set_id = config.scenario_set_id
    if scenario_manifest:
        scenario_set_id = str(scenario_manifest[0].get("scenario_set_id", scenario_set_id))
    return {
        "report_id": STAGE5_0D_REPORT_ID,
        "scenario_set_id": scenario_set_id,
        "candidate_tau_count": len(config.tau_candidates),
        "recommended_tau_candidate": None,
        "recommendation_basis": f"not_selected_stage5_0d_report_only:{source_kind}",
        "owner_decision_status": "required_not_provided",
        "owner_selected_tau": None,
        "owner_decision_note": "",
        "blocked_reason": (
            "final_tau_selection_requires_owner_decision_and_calibration_scope"
        ),
    }


def _checks(
    config: TauCalibrationReportConfig,
    topology_rows: list[dict[str, object]],
    source_kind: str,
) -> dict[str, object]:
    return {
        "candidate_tau_values_supplied": bool(config.tau_candidates),
        "candidate_tau_values_in_range": all(
            0.0 <= candidate.tau_candidate <= 1.0 for candidate in config.tau_candidates
        ),
        "no_default_tau_candidates": all(
            not candidate.is_default for candidate in config.tau_candidates
        ),
        "no_final_tau_selected": all(
            not candidate.is_selected for candidate in config.tau_candidates
        ),
        "stage4_8_threshold_not_used_as_default": all(
            not candidate.is_default for candidate in config.tau_candidates
        ),
        "source_is_smoke_test_only": source_kind == SOURCE_KIND_STAGE4_8,
        "source_is_alpha_fixture_suite": source_kind == SOURCE_KIND_STAGE5_0F,
        "metric_valued_fields_registered": True,
        "non_saturated_consensus_present": any(
            0.0 < float(row["consensus_success_probability"]) < 1.0 for row in topology_rows
        ),
        "sparse_better_than_full_graph_for_some_family": _sparse_better_than_full_graph(
            topology_rows
        ),
        "full_graph_not_oracle": all(
            not bool(row["is_oracle_candidate"])
            for row in topology_rows
            if bool(row["is_full_graph_baseline"])
        ),
        "oracle_labels_not_actor_inputs": all(
            not bool(row["is_deployment_actor_input"]) for row in topology_rows
        ),
        "reward_implemented": False,
        "training_run": False,
        "v5_code_migrated": False,
    }


def _source_scope(source_report: Mapping[str, object], source_kind: str) -> str:
    if source_kind == SOURCE_KIND_STAGE5_0F:
        return str(source_report.get("source_scope", "stage5_0f_alpha_fixture_suite"))
    return STAGE5_0D_SOURCE_SCOPE


def _rows_by_scenario_family(
    rows: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        result.setdefault(str(row["scenario_family"]), []).append(row)
    return dict(sorted(result.items()))


def _sparse_better_than_full_graph(topology_rows: list[dict[str, object]]) -> bool:
    for family_rows in _rows_by_scenario_family(topology_rows).values():
        sparse_rows = [
            row for row in family_rows if row["topology_family"] == "sparse_candidate"
        ]
        full_rows = [row for row in family_rows if bool(row["is_full_graph_baseline"])]
        for sparse_row in sparse_rows:
            for full_row in full_rows:
                if (
                    float(sparse_row["consensus_success_probability"]) > 0.0
                    and float(sparse_row["latency"]) < float(full_row["latency"])
                    and float(sparse_row["energy"]) < float(full_row["energy"])
                ):
                    return True
    return False


def _topology_family(row: Mapping[str, object], flags: tuple[object, ...]) -> str:
    if bool(row["is_full_graph_baseline"]):
        return "dense_full_graph_baseline"
    if "sparse_resource_efficient" in flags:
        return "sparse_candidate"
    if "failed_scheduled_message" in flags:
        return "failed_scheduled_message"
    if "synthetic_pbft_matrix" in flags:
        return "synthetic_pbft_diagnostic"
    if int(row["selected_edge_count"]) <= 1:
        return "sparse_candidate"
    return "communication_boundary_case"


def _minimum_or_none(rows: list[Mapping[str, object]], field: str) -> float | None:
    if not rows:
        return None
    return min(float(row[field]) for row in rows)


def _pareto_count(rows: list[Mapping[str, object]]) -> int:
    count = 0
    for row in rows:
        latency = float(row["latency"])
        energy = float(row["energy"])
        dominated = False
        for other in rows:
            if other is row:
                continue
            other_latency = float(other["latency"])
            other_energy = float(other["energy"])
            if (
                other_latency <= latency
                and other_energy <= energy
                and (other_latency < latency or other_energy < energy)
            ):
                dominated = True
                break
        if not dominated:
            count += 1
    return count


def _is_weak_primary_case(row: Mapping[str, object]) -> bool:
    flags = set(row.get("diagnostic_flags", ()))
    if "weak_edge_primary" in flags:
        return True
    per_primary = row.get("per_primary_reliability", {})
    if not isinstance(per_primary, Mapping) or len(per_primary) < 2:
        return False
    values = [float(value) for value in per_primary.values()]
    return max(values) - min(values) > 0.2


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("expected mapping payload")
    return value


def _sequence(value: object) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("expected sequence payload")
    return tuple(value)
