from marl_topology.evaluation import (
    REQUIRED_SWEEP_IDS,
    REQUIRED_SWEEP_ROW_FIELDS,
    STAGE5_0I_SWEEP_DESIGN_STAGE_ID,
    TAU_REQUIREMENT_MIN,
    build_stage5_0i_feasibility_envelope_sweep_design,
)


def test_stage5_0i_manifest_records_fixed_requirement_and_no_selection() -> None:
    design = build_stage5_0i_feasibility_envelope_sweep_design()

    assert design["stage"] == STAGE5_0I_SWEEP_DESIGN_STAGE_ID
    assert design["tau_requirement_min"] == TAU_REQUIREMENT_MIN == 0.9
    tau_policy = design["tau_policy"]
    assert tau_policy["fixed_during_sweeps"] is True
    assert tau_policy["lower_tau_diagnostic_values_allowed_in_sweep"] is False
    assert tau_policy["final_tau_selected"] is False
    assert tau_policy["final_tau_below_requirement_selected"] is False

    checks = design["checks"]
    assert checks["tau_requirement_min_fixed"] is True
    assert checks["final_tau_selected"] is False
    assert checks["final_tau_below_requirement_selected"] is False
    assert checks["simulation_parameters_changed_to_force_feasibility"] is False
    assert checks["reward_implemented"] is False
    assert checks["training_run"] is False
    assert checks["v5_code_migrated"] is False


def test_stage5_0i_required_sweep_axes_are_present_and_complete() -> None:
    design = build_stage5_0i_feasibility_envelope_sweep_design()
    axes = {axis["sweep_id"]: axis for axis in design["sweep_axes"]}

    assert set(REQUIRED_SWEEP_IDS).issubset(axes)
    for sweep_id in REQUIRED_SWEEP_IDS:
        axis = axes[sweep_id]
        assert axis["controlled_parameter"]
        assert axis["unit"]
        assert axis["target_failure_reasons"]
        assert axis["affected_fixture_families"]
        assert axis["expected_signal"]
        assert axis["monotonicity_expectation"]
        assert set(REQUIRED_SWEEP_ROW_FIELDS).issubset(axis["required_output_fields"])
        assert "do not lower tau_requirement_min" in axis["negative_controls"]
        assert "tau_requirement_min" in axis["held_constant"]


def test_stage5_0i_sweep_schema_and_metric_governance_are_bounded() -> None:
    design = build_stage5_0i_feasibility_envelope_sweep_design()

    schema_fields = {row["field"]: row for row in design["sweep_row_schema"]}
    for field in REQUIRED_SWEEP_ROW_FIELDS:
        assert field in schema_fields
        assert schema_fields[field]["required"] is True
        assert schema_fields[field]["actor_deployment_visible"] is False

    metric_governance = design["metric_governance"]
    assert metric_governance["metric_valued_fields"] == [
        "consensus_success_probability",
        "latency",
        "energy",
        "topology_diagnostics",
    ]
    assert metric_governance["new_metric_names_introduced"] == []
    assert schema_fields["consensus_success_probability"]["registered_metric_concept"] == (
        "consensus_success_probability"
    )
    assert schema_fields["latency"]["registered_metric_concept"] == "latency"
    assert schema_fields["energy"]["registered_metric_concept"] == "energy"
    assert schema_fields["is_deployment_actor_input"]["actor_deployment_visible"] is False


def test_stage5_0i_execution_order_and_comparison_policy_are_explicit() -> None:
    design = build_stage5_0i_feasibility_envelope_sweep_design()

    phases = [row["phase"] for row in design["sweep_execution_order"]]
    assert phases == [
        "single_axis_physics_and_link",
        "single_axis_timing_and_resource",
        "single_axis_geometry_and_topology",
        "protocol_conservatism_check",
    ]
    comparison = design["comparison_policy"]
    assert comparison["primary_feasibility_test"] == (
        "consensus_success_probability >= tau_requirement_min"
    )
    assert "not an oracle" in comparison["dominance_warning"]
    assert "resource_objective_context" in comparison
