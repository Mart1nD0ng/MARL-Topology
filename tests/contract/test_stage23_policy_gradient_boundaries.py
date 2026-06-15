from marl_topology.training.policy_gradient.pilot_runner import (
    Stage23PolicyGradientConfig,
    build_stage23_reward_surrogate_config,
    run_stage23_preflight,
    run_stage23_selected_physical_policy_gradient_pilot,
)


def test_stage23_preflight_confirms_selected_physical_actor_and_objective_stack() -> None:
    preflight = run_stage23_preflight(project_root=".")
    gates = preflight["gates"]

    assert preflight["preflight_passed"] is True
    assert gates["active_action_semantics_selected_physical"]["passed"] is True
    assert gates["discarded_action_semantics_not_active"]["passed"] is True
    assert gates["full_gnn_v2_actor_active"]["passed"] is True
    assert gates["actor_safe_input_boundary"]["passed"] is True
    assert gates["physical_link_assembler_active"]["passed"] is True
    assert gates["stage3_stage4_objective_evaluator_active"]["passed"] is True
    assert gates["reward_surrogate_config_read_only"]["passed"] is True
    assert gates["run_manifest_validator_available"]["passed"] is True


def test_stage23_reward_config_is_read_only_across_pilot() -> None:
    before = build_stage23_reward_surrogate_config()
    report = run_stage23_selected_physical_policy_gradient_pilot(
        config=Stage23PolicyGradientConfig(policy_updates=1),
        project_root=".",
    )
    after = build_stage23_reward_surrogate_config()

    assert report["reward_config_unchanged"] is True
    assert before == after
    assert before.tau == 0.9
    assert before.config_id == "stage5_3_surrogate_config_with_selected_references"
    # Active reward is the owner-approved feasibility-first barrier (not the flat sum).
    assert before.structure == "feasibility_first_barrier_v2"


def test_stage23_actor_and_sampler_boundaries_block_forbidden_inputs() -> None:
    preflight = run_stage23_preflight(project_root=".")
    boundary = preflight["gates"]["full_gnn_v2_actor_active"]["boundary_report"]
    forbidden = set(preflight["gates"]["actor_safe_input_boundary"]["forbidden_sampler_input_fields"])

    assert boundary["outputs_edge_scores_only"] is True
    assert boundary["global_topology_used"] is False
    assert boundary["critic_outputs_used"] is False
    assert "reward_surrogate" in forbidden
    assert "consensus_success_probability" in forbidden
    assert "latency" in forbidden
    assert "energy" in forbidden
