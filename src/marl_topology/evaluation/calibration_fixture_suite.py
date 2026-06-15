"""Executable alpha fixture suite for tau-consensus calibration reporting."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

from marl_topology.metrics import REGISTERED_METRICS, require_registered_metrics
from marl_topology.protocol import (
    FAULT_FILTER_NONE,
    PBFTExpectedInitiatorConfig,
    evaluate_expected_initiator_pbft_reliability,
)


STAGE5_0F_TAU_FIXTURE_SUITE_STAGE_ID = "stage_5_0f_minimal_tau_calibration_fixture_suite"
STAGE5_0F_SCENARIO_SET_ID = "stage5_0f_tau_alpha_suite"
STAGE5_0F_SOURCE_SCOPE = "stage5_0f_alpha_fixture_suite_not_tau_selection"
STAGE5_0F_NODE_IDS = ("center", "edge_a", "edge_b", "edge_c")
STAGE5_0F_FAULT_TOLERANCE = 1
STAGE5_0F_FAULT_FILTER_MODE = FAULT_FILTER_NONE
STAGE5_0F_REQUIRED_FAMILIES = (
    "clear_free_space_reference",
    "near_threshold_link_budget",
    "blocked_or_nlos_urban",
    "same_resource_interference",
    "deadline_tight_retransmission",
    "unreachable_reliability_target",
    "sparse_vs_dense_tradeoff",
    "weak_primary_distribution",
)
STAGE5_0F_REQUIRED_ROW_FIELDS = (
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
)
_REGISTERED_ROW_METRICS = (
    "consensus_success_probability",
    "latency",
    "energy",
    "topology_diagnostics",
)


@dataclass(frozen=True, slots=True)
class TauCalibrationFixtureRow:
    """One topology evaluation row in the Stage 5.0f alpha suite."""

    scenario_set_id: str
    scenario_family: str
    scenario_id: str
    fixture_id: str
    topology_name: str
    topology_family: str
    selected_edge_count: int
    is_full_graph_baseline: bool
    is_oracle_candidate: bool
    is_deployment_actor_input: bool
    consensus_success_probability: float
    latency: float
    energy: float
    per_primary_reliability: Mapping[str, float]
    diagnostic_flags: tuple[str, ...]
    coverage_axis: str

    def __post_init__(self) -> None:
        for field_name in (
            "scenario_set_id",
            "scenario_family",
            "scenario_id",
            "fixture_id",
            "topology_name",
            "topology_family",
            "coverage_axis",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if self.selected_edge_count < 0:
            raise ValueError("selected_edge_count must be nonnegative")
        if not 0.0 <= self.consensus_success_probability <= 1.0:
            raise ValueError("consensus_success_probability must be in [0, 1]")
        if self.latency < 0.0 or self.energy < 0.0:
            raise ValueError("latency and energy must be nonnegative")
        if set(self.per_primary_reliability) != set(STAGE5_0F_NODE_IDS):
            raise ValueError("per_primary_reliability must contain alpha suite node ids")
        if any(not 0.0 <= value <= 1.0 for value in self.per_primary_reliability.values()):
            raise ValueError("per-primary reliability values must be in [0, 1]")
        if self.is_full_graph_baseline and self.is_oracle_candidate:
            raise ValueError("full graph baseline must not be an oracle candidate")
        if self.is_deployment_actor_input:
            raise ValueError("fixture rows must not expose deployment actor inputs")

    def to_payload(self) -> dict[str, object]:
        payload = {
            "scenario_set_id": self.scenario_set_id,
            "scenario_family": self.scenario_family,
            "scenario_id": self.scenario_id,
            "fixture_id": self.fixture_id,
            "topology_name": self.topology_name,
            "topology_family": self.topology_family,
            "selected_edge_count": self.selected_edge_count,
            "is_full_graph_baseline": self.is_full_graph_baseline,
            "is_oracle_candidate": self.is_oracle_candidate,
            "is_deployment_actor_input": self.is_deployment_actor_input,
            "consensus_success_probability": self.consensus_success_probability,
            "latency": self.latency,
            "energy": self.energy,
            "per_primary_reliability": dict(self.per_primary_reliability),
            "diagnostic_flags": list(self.diagnostic_flags),
            "coverage_axis": self.coverage_axis,
        }
        missing = sorted(set(STAGE5_0F_REQUIRED_ROW_FIELDS) - set(payload))
        if missing:
            raise ValueError(f"fixture row missing required fields: {missing}")
        require_registered_metrics(_REGISTERED_ROW_METRICS)
        return payload


def build_stage5_0f_tau_calibration_fixture_suite_report() -> dict[str, object]:
    """Build the minimal executable alpha suite for tau report input."""

    rows = _build_rows()
    payload_rows = [row.to_payload() for row in rows]
    scenario_manifest = _scenario_manifest(rows)
    protocol_rows = [_protocol_regime_row()]
    return {
        "stage": STAGE5_0F_TAU_FIXTURE_SUITE_STAGE_ID,
        "scenario_set_id": STAGE5_0F_SCENARIO_SET_ID,
        "source_scope": STAGE5_0F_SOURCE_SCOPE,
        "scenario_manifest": scenario_manifest,
        "topology_evaluation_rows": payload_rows,
        "protocol_regime_rows": protocol_rows,
        "metric_governance": {
            "registered_metric_names": list(REGISTERED_METRICS.keys()),
            "metric_valued_fields": list(_REGISTERED_ROW_METRICS),
            "metric_valued_fields_registered": True,
            "new_metric_names_introduced": [],
        },
        "checks": _checks(payload_rows),
    }


def _build_rows() -> tuple[TauCalibrationFixtureRow, ...]:
    families = (
        _uniform_family(
            family_id="clear_free_space_reference",
            coverage_axis="geometry_visibility",
            weak_probability=0.0,
            sparse_probability=0.99,
            dense_probability=0.98,
            weak_latency=0.0,
            sparse_latency=0.0006,
            dense_latency=0.0014,
            weak_energy=0.0,
            sparse_energy=0.00011,
            dense_energy=0.00046,
            family_flags=("clear_free_space_reference", "los_reference"),
        ),
        _uniform_family(
            family_id="near_threshold_link_budget",
            coverage_axis="channel_link",
            weak_probability=0.75,
            sparse_probability=0.85,
            dense_probability=0.82,
            weak_latency=0.0012,
            sparse_latency=0.0010,
            dense_latency=0.0018,
            weak_energy=0.00024,
            sparse_energy=0.00020,
            dense_energy=0.00055,
            family_flags=("near_threshold_link_budget", "non_saturated_boundary"),
        ),
        _uniform_family(
            family_id="blocked_or_nlos_urban",
            coverage_axis="geometry_visibility",
            weak_probability=0.35,
            sparse_probability=0.82,
            dense_probability=0.70,
            weak_latency=0.0019,
            sparse_latency=0.0015,
            dense_latency=0.0023,
            weak_energy=0.00032,
            sparse_energy=0.00028,
            dense_energy=0.00068,
            family_flags=("blocked_or_nlos_urban", "nlos_penalty_visible"),
        ),
        _uniform_family(
            family_id="same_resource_interference",
            coverage_axis="network_resource",
            weak_probability=0.60,
            sparse_probability=0.99,
            dense_probability=0.75,
            weak_latency=0.0014,
            sparse_latency=0.00065,
            dense_latency=0.0026,
            weak_energy=0.00030,
            sparse_energy=0.00012,
            dense_energy=0.00074,
            family_flags=("same_resource_interference", "full_graph_interference_penalty"),
        ),
        _uniform_family(
            family_id="deadline_tight_retransmission",
            coverage_axis="deadline_retransmission",
            weak_probability=0.55,
            sparse_probability=0.90,
            dense_probability=0.82,
            weak_latency=0.00065,
            sparse_latency=0.00085,
            dense_latency=0.0017,
            weak_energy=0.00019,
            sparse_energy=0.00023,
            dense_energy=0.00062,
            family_flags=("deadline_tight_retransmission", "limited_attempt_budget"),
        ),
        _uniform_family(
            family_id="unreachable_reliability_target",
            coverage_axis="channel_link",
            weak_probability=0.0,
            sparse_probability=0.95,
            dense_probability=0.80,
            weak_latency=0.0021,
            sparse_latency=0.0011,
            dense_latency=0.0024,
            weak_energy=0.00042,
            sparse_energy=0.00026,
            dense_energy=0.00070,
            family_flags=("unreachable_reliability_target", "required_time_capped"),
        ),
        _uniform_family(
            family_id="sparse_vs_dense_tradeoff",
            coverage_axis="topology_tradeoff",
            weak_probability=0.50,
            sparse_probability=0.99,
            dense_probability=0.99,
            weak_latency=0.0018,
            sparse_latency=0.0007,
            dense_latency=0.0022,
            weak_energy=0.00036,
            sparse_energy=0.00013,
            dense_energy=0.00082,
            family_flags=("sparse_vs_dense_tradeoff", "sparse_resource_efficient"),
        ),
        _weak_primary_family(),
    )
    return tuple(row for family in families for row in family)


def _uniform_family(
    *,
    family_id: str,
    coverage_axis: str,
    weak_probability: float,
    sparse_probability: float,
    dense_probability: float,
    weak_latency: float,
    sparse_latency: float,
    dense_latency: float,
    weak_energy: float,
    sparse_energy: float,
    dense_energy: float,
    family_flags: tuple[str, ...],
) -> tuple[TauCalibrationFixtureRow, ...]:
    return (
        _row_from_uniform_probability(
            family_id=family_id,
            coverage_axis=coverage_axis,
            topology_variant="weak_baseline",
            topology_family="weak_or_disconnected_baseline",
            selected_edge_count=0 if weak_probability == 0.0 else 1,
            message_probability=weak_probability,
            latency=weak_latency,
            energy=weak_energy,
            flags=family_flags + ("weak_or_disconnected_baseline",),
        ),
        _row_from_uniform_probability(
            family_id=family_id,
            coverage_axis=coverage_axis,
            topology_variant="sparse_candidate",
            topology_family="sparse_candidate",
            selected_edge_count=1,
            message_probability=sparse_probability,
            latency=sparse_latency,
            energy=sparse_energy,
            flags=family_flags + ("sparse_candidate",),
        ),
        _row_from_uniform_probability(
            family_id=family_id,
            coverage_axis=coverage_axis,
            topology_variant="dense_full_graph_baseline",
            topology_family="dense_full_graph_baseline",
            selected_edge_count=6,
            message_probability=dense_probability,
            latency=dense_latency,
            energy=dense_energy,
            flags=family_flags + ("dense_full_graph_baseline",),
            is_full_graph_baseline=True,
        ),
    )


def _weak_primary_family() -> tuple[TauCalibrationFixtureRow, ...]:
    family_id = "weak_primary_distribution"
    coverage_axis = "pbft_primary_asymmetry"
    return (
        _row_from_matrix(
            family_id=family_id,
            coverage_axis=coverage_axis,
            topology_variant="weak_primary_baseline",
            topology_family="weak_or_disconnected_baseline",
            selected_edge_count=2,
            matrix=_sender_penalty_matrix(weak_sender="edge_a", weak_probability=0.55),
            latency=0.0013,
            energy=0.00031,
            flags=(family_id, "weak_edge_primary", "primary_spread_visible"),
        ),
        _row_from_matrix(
            family_id=family_id,
            coverage_axis=coverage_axis,
            topology_variant="sparse_candidate",
            topology_family="sparse_candidate",
            selected_edge_count=3,
            matrix=_sender_penalty_matrix(weak_sender="edge_a", weak_probability=0.78),
            latency=0.0010,
            energy=0.00024,
            flags=(family_id, "weak_primary_improved", "sparse_candidate"),
        ),
        _row_from_matrix(
            family_id=family_id,
            coverage_axis=coverage_axis,
            topology_variant="dense_full_graph_baseline",
            topology_family="dense_full_graph_baseline",
            selected_edge_count=6,
            matrix=_uniform_matrix(0.92),
            latency=0.0020,
            energy=0.00078,
            flags=(family_id, "balanced_dense_baseline", "dense_full_graph_baseline"),
            is_full_graph_baseline=True,
        ),
    )


def _row_from_uniform_probability(
    *,
    family_id: str,
    coverage_axis: str,
    topology_variant: str,
    topology_family: str,
    selected_edge_count: int,
    message_probability: float,
    latency: float,
    energy: float,
    flags: tuple[str, ...],
    is_full_graph_baseline: bool = False,
) -> TauCalibrationFixtureRow:
    return _row_from_matrix(
        family_id=family_id,
        coverage_axis=coverage_axis,
        topology_variant=topology_variant,
        topology_family=topology_family,
        selected_edge_count=selected_edge_count,
        matrix=_uniform_matrix(message_probability),
        latency=latency,
        energy=energy,
        flags=flags + (f"message_probability={message_probability:.2f}",),
        is_full_graph_baseline=is_full_graph_baseline,
    )


def _row_from_matrix(
    *,
    family_id: str,
    coverage_axis: str,
    topology_variant: str,
    topology_family: str,
    selected_edge_count: int,
    matrix: Mapping[tuple[str, str], float],
    latency: float,
    energy: float,
    flags: tuple[str, ...],
    is_full_graph_baseline: bool = False,
) -> TauCalibrationFixtureRow:
    reliability = evaluate_expected_initiator_pbft_reliability(
        PBFTExpectedInitiatorConfig(
            node_ids=STAGE5_0F_NODE_IDS,
            fault_tolerance=STAGE5_0F_FAULT_TOLERANCE,
            fault_filter_mode=STAGE5_0F_FAULT_FILTER_MODE,
        ),
        pre_prepare_matrix=matrix,
        prepare_matrix=matrix,
        commit_matrix=matrix,
    )
    probability = _checked_probability(reliability.consensus_success_probability)
    return TauCalibrationFixtureRow(
        scenario_set_id=STAGE5_0F_SCENARIO_SET_ID,
        scenario_family=family_id,
        scenario_id=f"{family_id}_scenario",
        fixture_id=f"{family_id}_alpha_fixture",
        topology_name=f"{family_id}/{topology_variant}",
        topology_family=topology_family,
        selected_edge_count=selected_edge_count,
        is_full_graph_baseline=is_full_graph_baseline,
        is_oracle_candidate=False,
        is_deployment_actor_input=False,
        consensus_success_probability=probability,
        latency=latency,
        energy=energy,
        per_primary_reliability=dict(reliability.per_primary_reliability),
        diagnostic_flags=flags + ("stage5_0f_alpha_suite",),
        coverage_axis=coverage_axis,
    )


def _uniform_matrix(probability: float) -> dict[tuple[str, str], float]:
    checked = _checked_probability(probability)
    return {
        (source_id, target_id): checked
        for source_id in STAGE5_0F_NODE_IDS
        for target_id in STAGE5_0F_NODE_IDS
        if source_id != target_id
    }


def _sender_penalty_matrix(
    *,
    weak_sender: str,
    weak_probability: float,
    strong_probability: float = 0.95,
) -> dict[tuple[str, str], float]:
    weak = _checked_probability(weak_probability)
    strong = _checked_probability(strong_probability)
    return {
        (source_id, target_id): weak if source_id == weak_sender else strong
        for source_id in STAGE5_0F_NODE_IDS
        for target_id in STAGE5_0F_NODE_IDS
        if source_id != target_id
    }


def _scenario_manifest(rows: tuple[TauCalibrationFixtureRow, ...]) -> list[dict[str, object]]:
    manifest: list[dict[str, object]] = []
    for family_id in STAGE5_0F_REQUIRED_FAMILIES:
        family_rows = tuple(row for row in rows if row.scenario_family == family_id)
        if not family_rows:
            raise ValueError(f"missing fixture family rows: {family_id}")
        first = family_rows[0]
        manifest.append(
            {
                "scenario_set_id": STAGE5_0F_SCENARIO_SET_ID,
                "scenario_family": family_id,
                "scenario_id": first.scenario_id,
                "fixture_id": first.fixture_id,
                "coverage_axis": first.coverage_axis,
                "physics_regime": "stage5_0f_alpha_fixture_declared_regime",
                "geometry_visibility_regime": "declared_by_family_diagnostic_flags",
                "channel_regime": "stage3_6_urlcc_finite_blocklength_v1",
                "link_transmission_regime": "urlcc_finite_blocklength_v1",
                "network_regime": "stage3_network_communication_v1",
                "pbft_model_id": "pbft_expected_initiator_mean_field_v1",
                "accounting_model_id": "stage4_protocol_scheduled_phase_occupancy_v1",
                "committee_size": len(STAGE5_0F_NODE_IDS),
                "fault_tolerance": STAGE5_0F_FAULT_TOLERANCE,
                "primary_distribution": "uniform",
                "fault_filter_mode": STAGE5_0F_FAULT_FILTER_MODE,
                "view_change_mode": "deferred",
                "deterministic_seed_policy": "deterministic_no_random_sampling",
                "topology_variant_ids": [row.topology_name for row in family_rows],
                "expected_diagnostic_flags": sorted(
                    {flag for row in family_rows for flag in row.diagnostic_flags}
                ),
                "owner_scope_note": (
                    "Stage 5.0f alpha fixture suite is executable calibration "
                    "input, but it does not choose tau_consensus."
                ),
            }
        )
    return manifest


def _protocol_regime_row() -> dict[str, object]:
    return {
        "node_count": len(STAGE5_0F_NODE_IDS),
        "fault_tolerance": STAGE5_0F_FAULT_TOLERANCE,
        "phase_budgets_s": "alpha_fixture_declared_scheduled_latency_rows",
        "primary_distribution": "uniform",
        "fault_filter_mode": STAGE5_0F_FAULT_FILTER_MODE,
        "view_change_mode": "deferred",
        "mean_field_assumption": True,
        "urlcc_finite_blocklength_v1": True,
        "pbft_expected_initiator_mean_field_v1": True,
    }


def _checks(rows: list[dict[str, object]]) -> dict[str, object]:
    family_ids = {str(row["scenario_family"]) for row in rows}
    variants_by_family = {
        family_id: [row for row in rows if row["scenario_family"] == family_id]
        for family_id in family_ids
    }
    full_rows = [row for row in rows if row["is_full_graph_baseline"]]
    sparse_rows = [row for row in rows if row["topology_family"] == "sparse_candidate"]
    return {
        "required_families_present": set(STAGE5_0F_REQUIRED_FAMILIES).issubset(family_ids),
        "minimum_three_variants_per_family": all(
            len(variants_by_family[family_id]) >= 3 for family_id in family_ids
        ),
        "required_fields_present": all(
            set(STAGE5_0F_REQUIRED_ROW_FIELDS).issubset(row) for row in rows
        ),
        "consensus_probabilities_in_range": all(
            0.0 <= float(row["consensus_success_probability"]) <= 1.0 for row in rows
        ),
        "latency_energy_nonnegative": all(
            float(row["latency"]) >= 0.0 and float(row["energy"]) >= 0.0 for row in rows
        ),
        "non_saturated_consensus_present": any(
            0.0 < float(row["consensus_success_probability"]) < 1.0 for row in rows
        ),
        "sparse_candidate_present": bool(sparse_rows),
        "full_graph_baseline_present": bool(full_rows),
        "sparse_better_than_full_graph_for_some_family": _sparse_better_than_full(rows),
        "full_graph_not_oracle": all(not bool(row["is_oracle_candidate"]) for row in full_rows),
        "oracle_labels_not_actor_inputs": all(
            not bool(row["is_deployment_actor_input"]) for row in rows
        ),
        "failed_scheduled_message_has_latency_or_energy": any(
            "unreachable_reliability_target" in row["diagnostic_flags"]
            and (float(row["latency"]) > 0.0 or float(row["energy"]) > 0.0)
            for row in rows
        ),
        "weak_primary_spread_present": any(_per_primary_spread(row) > 0.2 for row in rows),
        "no_final_tau_selected": True,
        "reward_implemented": False,
        "training_run": False,
        "v5_code_migrated": False,
    }


def _sparse_better_than_full(rows: list[dict[str, object]]) -> bool:
    family_ids = {str(row["scenario_family"]) for row in rows}
    for family_id in family_ids:
        sparse = [
            row
            for row in rows
            if row["scenario_family"] == family_id and row["topology_family"] == "sparse_candidate"
        ]
        full = [
            row
            for row in rows
            if row["scenario_family"] == family_id and row["is_full_graph_baseline"]
        ]
        for sparse_row in sparse:
            for full_row in full:
                if (
                    float(sparse_row["consensus_success_probability"]) > 0.0
                    and float(sparse_row["latency"]) < float(full_row["latency"])
                    and float(sparse_row["energy"]) < float(full_row["energy"])
                ):
                    return True
    return False


def _per_primary_spread(row: Mapping[str, object]) -> float:
    values = [float(value) for value in dict(row["per_primary_reliability"]).values()]
    return max(values) - min(values)


def _checked_probability(value: float) -> float:
    if not isfinite(value):
        raise ValueError("probability must be finite")
    if value < 0.0 and value > -1e-15:
        return 0.0
    if value > 1.0 and value < 1.0 + 1e-15:
        return 1.0
    if not 0.0 <= value <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    return value
