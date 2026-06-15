"""Stage 5.0i feasibility envelope sweep design manifest.

The manifest is executable as a design sensor only. It does not run sweeps,
change simulator parameters, select final tau, implement reward, or train
models.
"""

from __future__ import annotations

from marl_topology.metrics import REGISTERED_METRICS, require_registered_metrics

from .requirement_feasibility_diagnosis import (
    STAGE5_0H_REQUIREMENT_DIAGNOSIS_STAGE_ID,
    SUPPORTED_FAILURE_REASONS,
    TAU_REQUIREMENT_MIN,
)


STAGE5_0I_SWEEP_DESIGN_STAGE_ID = "stage_5_0i_feasibility_envelope_sweep_design"
REQUIRED_SWEEP_IDS = (
    "bandwidth_sweep",
    "tx_power_sweep",
    "deadline_sweep",
    "payload_sweep",
    "rsu_height_placement_sweep",
    "resource_orthogonalization_sweep",
    "fault_filter_mode_comparison",
    "topology_candidate_expansion",
)
REQUIRED_SWEEP_ROW_FIELDS = (
    "sweep_id",
    "sweep_run_id",
    "scenario_family",
    "fixture_id",
    "topology_name",
    "controlled_parameter",
    "control_value_label",
    "control_unit",
    "tau_requirement_min",
    "requirement_met",
    "consensus_success_probability",
    "latency",
    "energy",
    "failure_reason_before",
    "failure_reason_after",
    "selected_edge_count",
    "is_full_graph_baseline",
    "is_oracle_candidate",
    "is_deployment_actor_input",
    "parameter_source",
    "diagnostic_flags",
)
REGISTERED_SWEEP_METRIC_FIELDS = (
    "consensus_success_probability",
    "latency",
    "energy",
    "topology_diagnostics",
)


def build_stage5_0i_feasibility_envelope_sweep_design() -> dict[str, object]:
    """Return the Stage 5.0i sweep-design manifest without running a sweep."""

    require_registered_metrics(REGISTERED_SWEEP_METRIC_FIELDS)
    sweep_axes = _sweep_axes()
    return {
        "stage": STAGE5_0I_SWEEP_DESIGN_STAGE_ID,
        "source_stage": STAGE5_0H_REQUIREMENT_DIAGNOSIS_STAGE_ID,
        "tau_requirement_min": TAU_REQUIREMENT_MIN,
        "tau_policy": {
            "requirement_baseline": TAU_REQUIREMENT_MIN,
            "fixed_during_sweeps": True,
            "lower_tau_diagnostic_values_allowed_in_sweep": False,
            "final_tau_selected": False,
            "final_tau_below_requirement_selected": False,
        },
        "design_scope": (
            "design-only manifest for future minimal executable feasibility "
            "envelope sweeps"
        ),
        "controlled_object": (
            "Stage 3 communication resources, Stage 4 PBFT reliability controls, "
            "and topology candidate generation under a fixed reliability "
            "requirement"
        ),
        "design_rules": _design_rules(),
        "sweep_axes": sweep_axes,
        "required_sweep_ids": list(REQUIRED_SWEEP_IDS),
        "sweep_execution_order": _execution_order(),
        "sweep_row_schema": _sweep_row_schema(),
        "comparison_policy": _comparison_policy(),
        "acceptance_sensors": _acceptance_sensors(),
        "forbidden_operations": _forbidden_operations(),
        "metric_governance": {
            "registered_metric_names": list(REGISTERED_METRICS.keys()),
            "metric_valued_fields": list(REGISTERED_SWEEP_METRIC_FIELDS),
            "new_metric_names_introduced": [],
            "diagnostic_fields_not_primary_objectives": [
                "failure_reason_before",
                "failure_reason_after",
                "diagnostic_flags",
                "control_value_label",
                "parameter_source",
            ],
        },
        "checks": _checks(sweep_axes),
    }


def _design_rules() -> list[str]:
    return [
        "hold tau_requirement_min at 0.9 for every sweep row",
        "vary one control family at a time before combined sweeps",
        "keep baseline scenario topology and random seed policy fixed within a sweep",
        "compare against the unswept baseline row before interpreting feasibility",
        "report consensus_success_probability latency energy and diagnostics separately",
        "do not use lower tau diagnostic values as final thresholds",
        "do not treat full graph as an oracle or resource optimum",
        "do not expose oracle labels or sweep labels to deployment actor inputs",
        "do not implement reward or train models during envelope diagnosis",
        "do not change parameters merely to force feasibility",
    ]


def _sweep_axes() -> list[dict[str, object]]:
    return [
        _axis(
            sweep_id="bandwidth_sweep",
            controlled_parameter="bandwidth_hz",
            unit="Hz",
            actuator_family="communication_resource",
            target_failure_reasons=("link_budget_failure", "retransmission_insufficient"),
            affected_fixture_families=(
                "near_threshold_link_budget",
                "blocked_or_nlos_urban",
                "deadline_tight_retransmission",
            ),
            control_value_policy="baseline plus bounded higher values from Stage 3 configuration",
            expected_signal="higher bandwidth should increase reliability or reduce required transmission duration",
            monotonicity_expectation="nondecreasing consensus_success_probability when other controls are fixed",
        ),
        _axis(
            sweep_id="tx_power_sweep",
            controlled_parameter="tx_power_w",
            unit="W",
            actuator_family="communication_resource",
            target_failure_reasons=("link_budget_failure",),
            affected_fixture_families=("near_threshold_link_budget", "blocked_or_nlos_urban"),
            control_value_policy="baseline plus bounded higher values within regulatory assumptions",
            expected_signal="higher transmit power should improve SINR-limited link reliability",
            monotonicity_expectation="nondecreasing link and consensus reliability until saturation",
        ),
        _axis(
            sweep_id="deadline_sweep",
            controlled_parameter="deadline_s",
            unit="s",
            actuator_family="timing_budget",
            target_failure_reasons=("deadline_failure", "retransmission_insufficient"),
            affected_fixture_families=("deadline_tight_retransmission",),
            control_value_policy="baseline plus bounded longer deadlines from application latency budget",
            expected_signal="longer deadline should increase allowed attempts and deadline delivery probability",
            monotonicity_expectation="nondecreasing reliability with latency cost reported separately",
        ),
        _axis(
            sweep_id="payload_sweep",
            controlled_parameter="payload_bits",
            unit="bits",
            actuator_family="message_size",
            target_failure_reasons=("link_budget_failure", "deadline_failure", "resource_budget_failure"),
            affected_fixture_families=(
                "near_threshold_link_budget",
                "deadline_tight_retransmission",
                "unreachable_reliability_target",
            ),
            control_value_policy="baseline plus bounded smaller and larger payload labels",
            expected_signal="smaller payload should reduce required duration and improve feasibility",
            monotonicity_expectation="nonincreasing reliability as payload grows when other controls are fixed",
        ),
        _axis(
            sweep_id="rsu_height_placement_sweep",
            controlled_parameter="rsu_height_and_position",
            unit="m",
            actuator_family="geometry_visibility",
            target_failure_reasons=("link_budget_failure", "topology_candidate_failure"),
            affected_fixture_families=("blocked_or_nlos_urban",),
            control_value_policy="baseline plus explicit RSU height and placement alternatives",
            expected_signal="higher or better-placed RSUs should recover LoS corridors in blocked scenarios",
            monotonicity_expectation="visibility should improve only when geometry actually clears blockers",
        ),
        _axis(
            sweep_id="resource_orthogonalization_sweep",
            controlled_parameter="channel_resource_assignment",
            unit="resource_label",
            actuator_family="network_resource",
            target_failure_reasons=("interference_failure", "resource_budget_failure"),
            affected_fixture_families=("same_resource_interference", "sparse_vs_dense_tradeoff"),
            control_value_policy="shared-resource baseline vs orthogonalized channel labels",
            expected_signal="orthogonal resources should improve dense/full rows hurt by interference",
            monotonicity_expectation="interference should not increase under orthogonal resource assignment",
        ),
        _axis(
            sweep_id="fault_filter_mode_comparison",
            controlled_parameter="fault_filter_mode",
            unit="mode",
            actuator_family="protocol_reliability",
            target_failure_reasons=("pbft_quorum_failure", "primary_distribution_failure"),
            affected_fixture_families=("weak_primary_distribution",),
            control_value_policy="compare none and remove_largest without claiming strict Byzantine adversary model",
            expected_signal="remove_largest should lower or preserve reliability relative to none",
            monotonicity_expectation="conservative filter must not increase consensus_success_probability",
        ),
        _axis(
            sweep_id="topology_candidate_expansion",
            controlled_parameter="candidate_edge_radius_and_candidate_edges",
            unit="m_or_edge_count",
            actuator_family="topology_candidate_set",
            target_failure_reasons=("topology_candidate_failure", "primary_distribution_failure"),
            affected_fixture_families=(
                "clear_free_space_reference",
                "sparse_vs_dense_tradeoff",
                "weak_primary_distribution",
            ),
            control_value_policy="baseline candidate set plus bounded radius or extra edge candidates",
            expected_signal="expanded candidates should improve weak topology failures if communication resources suffice",
            monotonicity_expectation="candidate availability can improve feasibility but may raise latency and energy",
        ),
    ]


def _axis(
    *,
    sweep_id: str,
    controlled_parameter: str,
    unit: str,
    actuator_family: str,
    target_failure_reasons: tuple[str, ...],
    affected_fixture_families: tuple[str, ...],
    control_value_policy: str,
    expected_signal: str,
    monotonicity_expectation: str,
) -> dict[str, object]:
    unsupported = sorted(set(target_failure_reasons) - set(SUPPORTED_FAILURE_REASONS))
    if unsupported:
        raise ValueError(f"unsupported failure reasons for sweep axis: {unsupported}")
    return {
        "sweep_id": sweep_id,
        "controlled_parameter": controlled_parameter,
        "unit": unit,
        "actuator_family": actuator_family,
        "target_failure_reasons": list(target_failure_reasons),
        "affected_fixture_families": list(affected_fixture_families),
        "control_value_policy": control_value_policy,
        "expected_signal": expected_signal,
        "monotonicity_expectation": monotonicity_expectation,
        "held_constant": [
            "tau_requirement_min",
            "objective feasibility direction",
            "PBFT model id unless fault_filter_mode is the controlled parameter",
            "scenario seed and base topology unless topology_candidate_expansion is controlled",
            "reward/training/model absence",
        ],
        "minimum_rows": [
            "baseline",
            "single_control_low_or_reference",
            "single_control_high_or_stress",
        ],
        "required_output_fields": list(REQUIRED_SWEEP_ROW_FIELDS),
        "negative_controls": [
            "do not lower tau_requirement_min",
            "do not infer infeasibility from a failed policy",
            "do not make full graph an oracle",
            "do not expose sweep labels to deployment actor inputs",
        ],
    }


def _execution_order() -> list[dict[str, object]]:
    return [
        {
            "phase": "single_axis_physics_and_link",
            "sweep_ids": ["bandwidth_sweep", "tx_power_sweep", "payload_sweep"],
            "reason": "resolve link-budget and finite-blocklength observability gaps first",
        },
        {
            "phase": "single_axis_timing_and_resource",
            "sweep_ids": ["deadline_sweep", "resource_orthogonalization_sweep"],
            "reason": "separate deadline and interference failures from topology failures",
        },
        {
            "phase": "single_axis_geometry_and_topology",
            "sweep_ids": ["rsu_height_placement_sweep", "topology_candidate_expansion"],
            "reason": "test whether geometry and candidate-set actuators can restore feasibility",
        },
        {
            "phase": "protocol_conservatism_check",
            "sweep_ids": ["fault_filter_mode_comparison"],
            "reason": "measure optimistic vs conservative PBFT reliability sensitivity",
        },
    ]


def _sweep_row_schema() -> list[dict[str, object]]:
    metric_fields = set(REGISTERED_SWEEP_METRIC_FIELDS)
    rows = []
    for field in REQUIRED_SWEEP_ROW_FIELDS:
        rows.append(
            {
                "field": field,
                "registered_metric_concept": field if field in metric_fields else None,
                "required": True,
                "actor_deployment_visible": False,
            }
        )
    return rows


def _comparison_policy() -> dict[str, object]:
    return {
        "baseline": "unswept Stage 5.0h row or Stage 3-backed equivalent",
        "primary_feasibility_test": "consensus_success_probability >= tau_requirement_min",
        "resource_objective_context": "compare latency and energy only after feasibility status is known",
        "dominance_warning": "a feasible dense/full graph with higher latency and energy is not an oracle",
        "interpretation_order": [
            "did the controlled parameter move the intended failure reason",
            "did reliability reach tau_requirement_min",
            "what latency and energy cost changed",
            "did failure_reason_after change to a more specific cause",
            "is the effect consistent with monotonicity expectation",
        ],
    }


def _acceptance_sensors() -> list[str]:
    return [
        "all required sweep ids are present",
        "each sweep row includes registered metric-valued fields only",
        "each sweep holds tau_requirement_min fixed at 0.9",
        "each sweep varies one control family at a time",
        "monotonic sanity checks pass or produce explicit anomaly flags",
        "full graph remains a baseline and not an oracle",
        "oracle and sweep labels stay out of deployment actor inputs",
        "reward implementation and training remain absent",
    ]


def _forbidden_operations() -> list[str]:
    return [
        "selecting final tau_consensus",
        "lowering tau_requirement_min below 0.9",
        "using lower diagnostic tau values as final thresholds",
        "changing simulation parameters only to force feasibility",
        "implementing reward or reward weights",
        "running training or adding model code",
        "migrating v5 code or phase scripts",
        "treating full graph as oracle",
    ]


def _checks(sweep_axes: list[dict[str, object]]) -> dict[str, object]:
    sweep_ids = {str(axis["sweep_id"]) for axis in sweep_axes}
    required_fields = set(REQUIRED_SWEEP_ROW_FIELDS)
    return {
        "tau_requirement_min_fixed": TAU_REQUIREMENT_MIN == 0.9,
        "final_tau_selected": False,
        "final_tau_below_requirement_selected": False,
        "required_sweeps_present": set(REQUIRED_SWEEP_IDS).issubset(sweep_ids),
        "one_control_family_per_sweep": all(
            isinstance(axis["controlled_parameter"], str)
            and bool(axis["controlled_parameter"])
            for axis in sweep_axes
        ),
        "required_output_fields_present": all(
            required_fields.issubset(set(axis["required_output_fields"]))
            for axis in sweep_axes
        ),
        "registered_metric_fields_only": True,
        "lower_tau_diagnostic_values_not_sweep_thresholds": True,
        "full_graph_not_oracle": True,
        "oracle_labels_not_actor_inputs": True,
        "simulation_parameters_changed_to_force_feasibility": False,
        "reward_implemented": False,
        "training_run": False,
        "v5_code_migrated": False,
    }
