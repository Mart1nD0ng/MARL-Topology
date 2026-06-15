# Project State

This file records the current project-control state for MARL-Topology. It is a decision aid, not authorization for Codex to start the next stage.

## State Snapshot

```yaml
current_stage: post_stage_34_gnn_ablation_blocked_awaiting_owner_decision
historical_current_stage_markers:
  - "current_stage: post_stage_33_gnn_repair_blocked_awaiting_owner_decision"
  - "current_stage: stage32a_complete_real_mappo_honest_backward_larger_graphs_gnn_competitive_but_unstable"
  - "current_stage: stage32a_real_mappo_loss_honest_backward_larger_graphs_run_in_progress"
  - "current_stage: stage32_execution_complete_gnn_actor_did_not_beat_mlp_mlp_remains_production_actor"
  - "current_stage: stage32_production_training_design_contract_drafted_awaiting_owner_decision_for_execution"
  - "current_stage: stage31_complete_production_readiness_verified_awaiting_owner_decision_for_scale_up"
  - "current_stage: stage31_production_readiness_unfreeze_and_implementation_in_progress"
  - "current_stage: post_stage30_repair_loop_blocked_awaiting_owner_decision"
  - "current_stage: post_stage_29_complete_pre_scale_decision_review_awaiting_owner_decision"
  - "current_stage: post_stage_28_complete_repaired_critic_mappo_rerun_awaiting_owner_decision"
  - "current_stage: post_stage_27_complete_critic_repaired_awaiting_owner_decision"
  - "current_stage: post_stage_26_complete_full_system_health_diagnostic_awaiting_owner_decision"
  - "current_stage: post_stage_25_complete_small_scale_mappo_pilot_awaiting_owner_decision"
  - "current_stage: post_stage_24_complete_mappo_micro_loop_ready_for_owner_decision"
  - "current_stage: post_stage_23_complete_pg_pilot_landed_awaiting_owner_decision"
  - "current_stage: post_stage_23_readiness_review_complete_policy_gradient_blocked_awaiting_owner_decision"
  - "current_stage: post_stage_22_complete_awaiting_owner_decision_for_stage_23"
  - "current_stage: post_stage_21_blocked_awaiting_owner_decision"
  - "current_stage: post_stage_20_complete_awaiting_owner_decision_for_stage_21"
  - "current_stage: post_stage_19_complete_awaiting_owner_decision_for_stage_20"
  - "current_stage: post_stage_18_complete_awaiting_owner_decision_for_stage_19"
  - "current_stage: post_stage_17_complete_awaiting_owner_decision_for_stage_18"
  - "current_stage: post_stage_16_complete_awaiting_owner_decision_for_stage_17"
previous_stage_aliases:
  - post_stage_34_gnn_ablation_blocked_awaiting_owner_decision
  - post_stage_33_gnn_repair_blocked_awaiting_owner_decision
  - post_stage_29_complete_pre_scale_decision_review_awaiting_owner_decision
  - post_stage_28_complete_repaired_critic_mappo_rerun_awaiting_owner_decision
  - post_stage_27_complete_critic_repaired_awaiting_owner_decision
  - post_stage_26_complete_full_system_health_diagnostic_awaiting_owner_decision
  - post_stage_25_complete_small_scale_mappo_pilot_awaiting_owner_decision
  - post_stage_24_complete_mappo_micro_loop_ready_for_owner_decision
  - post_stage_23_complete_pg_pilot_landed_awaiting_owner_decision
  - post_stage_23_readiness_review_complete_policy_gradient_blocked_awaiting_owner_decision
  - post_stage_22_complete_awaiting_owner_decision_for_stage_23
  - post_stage_21_blocked_awaiting_owner_decision
  - post_stage_20_complete_awaiting_owner_decision_for_stage_21
  - post_stage_19_complete_awaiting_owner_decision_for_stage_20
  - post_stage_18_complete_awaiting_owner_decision_for_stage_19
  - post_stage_17_complete_awaiting_owner_decision_for_stage_18
  - post_stage_16_complete_awaiting_owner_decision_for_stage_17
  - post_stage_15_complete_awaiting_owner_decision_for_stage_16
  - post_stage_9_0_local_mlp_edge_scorer_scaffold_awaiting_owner_decision
  - post_stage_8_complete_awaiting_owner_decision_for_stage_9
  - post_stage_8_0_awaiting_owner_decision_for_stage_8_1
  - post_stage_7_completion_closed_awaiting_owner_decision_for_stage_8
  - post_stage_7_1_stage_7_closed_awaiting_owner_decision_for_stage_8
  - post_stage_7_0_awaiting_owner_decision_for_stage_7_1
  - post_stage_6_closed_awaiting_owner_decision_for_stage_7
  - post_stage_6_0_awaiting_owner_decision_for_stage_6_1
  - post_stage_5_10_stage_5_closed_awaiting_owner_decision_for_stage_6
  - post_stage_5_9_awaiting_owner_decision_for_stage_5_10
  - post_stage_5_8_awaiting_owner_decision_for_stage_5_9
  - post_stage_5_7_awaiting_owner_decision_for_stage_5_8
  - post_stage_5_6_awaiting_owner_decision_for_stage_5_7
  - post_stage_5_5_awaiting_owner_decision_for_stage_5_6
  - post_stage_5_4_awaiting_owner_decision_for_stage_5_5
  - post_stage_5_3_awaiting_owner_decision_for_stage_5_4
  - post_stage_5_2_awaiting_owner_decision_for_stage_5_3
  - post_stage_5_1_awaiting_owner_decision_for_stage_5_2
  - post_stage_5_0m_ready_for_stage_5_1_owner_decision
  - post_stage_5_0l_awaiting_owner_decision
  - post_stage_5_0k_awaiting_owner_decision
  - post_stage_5_0j_awaiting_owner_decision
  - post_stage_5_0i_awaiting_owner_decision
  - post_stage_5_0h_awaiting_owner_decision
  - post_stage_5_0g_awaiting_owner_decision
  - post_stage_5_0f_awaiting_owner_decision
  - post_stage_5_0e_awaiting_owner_decision
  - post_stage_5_0d_awaiting_owner_decision
  - post_stage_5_0c_awaiting_owner_decision
  - post_stage_5_0a_awaiting_owner_decision
  - post_stage_5_0_awaiting_owner_decision
  - post_stage_4_8_awaiting_owner_decision
  - post_stage_4_7_awaiting_owner_decision
completed_stages:
  - stage_0_1_clean_core_scaffold
  - stage_0_2_metric_governance_reset
  - stage_1_v5_learning_audit
  - stage_1_1_lesson_to_gate_and_hygiene
  - stage_2_goal_skeleton_v0
  - stage_2_0_self_review_loop_upgrade
  - stage_2_1_dec_pomdp_schema_contract
  - stage_2_2_decentralized_non_learning_baselines
  - stage_2_3_minimal_dec_pomdp_env_wrapper
  - stage_2_4_baseline_evaluation_report
  - stage_2_5_scenario_fixture_contract
  - stage_2_6_replay_dataset_column_contract
  - stage_2_7_link_model_regime_review
  - stage_2_8_reward_contract_review_without_implementation
  - stage_2_8_protocol_timeout_review
  - stage_3_0_communication_simulation_design
  - stage_3_1_geometry_visibility_implementation
  - stage_3_2_channel_model_v1
  - stage_3_3_link_transmission_v1
  - stage_3_4_network_layer_communication_v1
  - stage_3_5_micro_fixture_suite_review
  - stage_3_6_urlcc_finite_blocklength_link_reliability
  - stage_4_0_pbft_application_consensus_contract_planning
  - stage_4_1_heterogeneous_quorum_tail_utility
  - stage_4_2_pbft_three_phase_reliability_record
  - stage_4_3_stage3_message_matrix_adapter
  - stage_4_4_expected_initiator_pbft_reliability
  - stage_4_5_baseline_and_oracle_review
  - stage_4_6_protocol_latency_energy_accounting_review
  - stage_4_7_pbft_application_evaluation_report
  - stage_4_8_communication_consensus_boundary_audit
  - stage_5_0_reward_objective_contract_freeze
  - stage_5_0a_tau_consensus_calibration_plan
  - stage_5_0c_tau_consensus_calibration_report_design
  - stage_5_0d_tau_consensus_calibration_report_implementation_without_tau_selection
  - stage_5_0e_tau_consensus_fixture_family_design
  - stage_5_0f_minimal_tau_calibration_fixture_suite
  - stage_5_0g_tau_consensus_calibration_report_run_with_owner_supplied_candidates_without_selection
  - stage_5_0h_requirement_anchored_feasibility_diagnosis
  - stage_5_0i_feasibility_envelope_sweep_design
  - stage_5_0j_minimal_executable_feasibility_envelope_sweep
  - stage_5_0k_stage3_backed_feasibility_envelope_sweep_hardening
  - stage_5_0l_stage3_backed_sweep_range_expansion_and_realism_review
  - stage_5_0m_objective_readiness_review_before_reward_implementation
  - stage_5_1_reward_implementation_plan_without_code
  - stage_5_2_reward_surrogate_interface_skeleton_with_contract_tests
  - stage_5_3_reward_normalization_reference_selection
  - stage_5_4_reward_report_integration_without_training
  - stage_5_5_training_preflight_review_without_training
  - stage_5_6_training_design_contract_without_execution
  - stage_5_7_policy_architecture_contract_without_implementation
  - stage_5_8_learning_target_replay_contract_without_implementation
  - stage_5_9_training_run_manifest_artifact_contract_without_execution
  - stage_5_10_run_manifest_validator_exit_gate_without_execution
  - stage_6_0_minimal_training_stack_implementation_with_manifest_guard
  - stage_6_1_actor_safe_batch_builder_without_model_or_training
  - stage_6_completion_exit_review
  - stage_7_0_learning_evidence_dataset_generation_with_owner_approval
  - stage_7_1_learning_evidence_data_quality_report_with_owner_approval
  - stage_7_completion_learning_evidence_dataset_build_quality_exit_gate
  - stage_7_completion_exit_review
  - stage_8_0_actor_policy_interface_contract_with_owner_approval
  - stage_8_policy_architecture_and_topology_assembler_deployment
  - stage_9_0_local_mlp_edge_scorer_baseline_without_training_execution
  - stage_10_supervised_loss_dry_run
  - stage_11_supervised_mlp_actor_warm_start
  - stage_12_critic_edge_delta_pretraining
  - stage_13_local_gnn_edge_scorer_comparison
  - stage_14_gru_lstm_temporal_actor_ablation
  - stage_15_controlled_policy_gradient_pilot
  - stage_9_to_stage_15_model_stack_goal_review
  - stage_16_learning_evidence_quality_improvement
  - stage_17_actor_observability_and_label_disambiguation
  - stage_18_evidence_rebuild_with_disambiguated_features_and_targets
  - stage_19_rerun_supervised_actor_stack_on_disambiguated_evidence
  - stage_20_supervised_actor_policy_evaluation_with_environment_assembler
  - stage_22_action_semantics_ab_full_gnn_repair
  - stage_23_controlled_policy_gradient_pilot_readiness_review
  - stage_23_selected_physical_policy_gradient_landing
  - stage_24_critic_integrated_mappo_micro_loop
  - stage_25_small_scale_formal_mappo_training_pilot
  - stage_26_full_system_health_diagnostic
  - stage_27_critic_baseline_repair
  - stage_28_rerun_small_scale_mappo_with_repaired_critic
  - stage_29_pre_scale_decision_review
  - stage30_closed_loop_repair_until_scale_readiness
  - stage31_phase_a_owner_decision_unfreeze_and_tau_feasibility_strategy
  - stage31_phase_b_procedural_scenario_generator_with_tau_feasibility_gradient
  - stage31_phase_c_feasibility_first_reward_surrogate_and_recalibration
  - stage31_phase_d_budget_aware_sampler_and_constraint_aware_actor_features
  - stage31_phase_e_scalable_teacher_labels_and_leakage_checked_split
  - stage31_phase_f_large_scale_readiness_test_all_blockers_resolved
  - stage32_production_training_design_contract_and_decision_packet_without_execution
  - stage32_production_training_execution_gnn_actor_repaired_critic_scaled_data_2000_5seed
  - stage32a_real_mappo_loss_honest_backward_larger_graphs_and_training_authorized_unfreeze
  - stage33_gnn_stability_and_mappo_loop_unification_failed_gate
  - stage34_gnn_baseline_repair_and_ablation_diagnostics_blocked_gate
active_gates:
  - metric_governance_gate
  - consensus_protocol_naming_gate
  - full_mask_not_oracle_gate
  - oracle_before_infeasible_gate
  - phase_script_entropy_gate
  - physics_regime_declaration_gate
  - dec_pomdp_leakage_gate
  - fixed_threshold_is_baseline_gate
  - baseline_evaluation_report_gate
  - scenario_fixture_contract_gate
  - replay_dataset_column_gate
  - link_model_regime_gate
  - reward_contract_review_gate
  - reward_plateau_resource_gate
  - protocol_timeout_gate
  - stage3_communication_simulation_gate
  - stage3_geometry_visibility_gate
  - stage3_channel_model_gate
  - stage3_link_transmission_gate
  - stage3_network_layer_gate
  - stage3_micro_fixture_suite_gate
  - stage3_6_urlcc_finite_blocklength_gate
  - stage4_pbft_application_consensus_plan_gate
  - stage4_heterogeneous_quorum_tail_utility_gate
  - stage4_pbft_three_phase_reliability_record_gate
  - stage4_stage3_message_matrix_adapter_gate
  - stage4_4_expected_initiator_pbft_gate
  - stage4_5_baseline_oracle_review_gate
  - stage4_6_protocol_accounting_gate
  - stage4_7_application_evaluation_report_gate
  - stage4_8_boundary_audit_gate
  - stage5_reward_objective_contract_freeze_gate
  - stage5_0a_tau_consensus_calibration_plan_gate
  - stage5_0c_tau_consensus_calibration_report_design_gate
  - stage5_0d_tau_consensus_calibration_report_gate
  - stage5_0e_tau_consensus_fixture_family_design_gate
  - stage5_0f_tau_calibration_fixture_suite_gate
  - stage5_0g_tau_consensus_calibration_report_run_gate
  - stage5_0h_requirement_feasibility_diagnosis_gate
  - stage5_0i_feasibility_envelope_sweep_design_gate
  - stage5_0j_minimal_feasibility_envelope_sweep_gate
  - stage5_0k_stage3_backed_feasibility_envelope_sweep_gate
  - stage5_0l_stage3_backed_sweep_range_review_gate
  - stage5_0m_objective_readiness_review_gate
  - stage5_1_reward_implementation_plan_gate
  - stage5_2_reward_surrogate_interface_gate
  - stage5_3_reward_normalization_reference_gate
  - stage5_4_reward_report_integration_gate
  - stage5_5_training_preflight_review_gate
  - stage5_6_training_design_contract_gate
  - stage5_7_policy_architecture_contract_gate
  - stage5_8_learning_target_replay_contract_gate
  - stage5_9_training_run_manifest_artifact_contract_gate
  - stage5_10_run_manifest_validator_exit_gate
  - stage_closure_discipline_gate
  - stage6_0_minimal_training_stack_guard
  - stage6_1_actor_safe_batch_builder_gate
  - stage6_completion_exit_gate
  - stage7_0_learning_evidence_dataset_gate
  - stage7_1_learning_evidence_quality_gate
  - stage7_completion_learning_evidence_exit_gate
  - stage7_completion_exit_gate
  - stage8_0_actor_policy_interface_contract_gate
  - stage8_policy_architecture_and_assembler_gate
  - stage8_topology_assembler_gate
  - stage8_actor_critic_interface_separation_gate
  - stage8_no_model_training_checkpoint_gate
  - stage9_0_active_edge_score_schema_gate
  - stage9_0_local_mlp_scaffold_gate
  - stage9_0_no_training_execution_gate
  - stage9_0_no_checkpoint_artifact_gate
  - stage10_supervised_loss_dry_run_gate
  - stage11_supervised_mlp_warm_start_gate
  - stage12_critic_edge_delta_pretraining_gate
  - stage13_local_gnn_comparison_gate
  - stage14_gru_lstm_temporal_ablation_gate
  - stage15_controlled_policy_gradient_pilot_gate
  - stage16_learning_evidence_quality_gate
  - stage17_actor_observability_label_disambiguation_gate
  - stage18_disambiguated_evidence_rebuild_gate
  - stage18_actor_safe_feature_gate
  - stage18_actor_target_rebuild_gate
  - stage18_critic_target_separation_gate
  - stage19_supervised_actor_stack_rerun_gate
  - stage19_no_policy_gradient_checkpoint_artifact_gate
  - stage20_environment_assembler_policy_evaluation_gate
  - stage20_policy_gradient_readiness_failed_gate
  - stage20_no_policy_gradient_checkpoint_artifact_gate
  - stage21_objective_stack_alignment_gate
  - stage21_actor_target_quality_gate
  - stage21_fair_assembler_evaluation_gate
  - stage21_projection_mismatch_improved_gate
  - stage21_actor_performance_failed_gate
  - stage21_boundary_safety_gate
  - stage21_policy_gradient_conditional_gate_failed
  - stage22_action_semantics_selection_gate
  - stage22_losing_semantics_cleanup_gate
  - stage22_full_message_passing_gnn_gate
  - stage22_objective_aware_teacher_gate
  - stage22_fair_same_assembler_evaluation_gate
  - stage22_policy_gradient_still_blocked_gate
  - stage23_low_entropy_preflight_cleanup_gate
  - stage23_single_selected_action_semantics_gate
  - stage23_full_message_passing_gnn_active_gate
  - stage23_objective_stack_selected_evidence_gate
  - stage23_selected_physical_policy_gradient_landing_gate
  - stage23_sampler_ab_c_trial_gate
  - stage23_low_entropy_sampler_promotion_gate
  - stage23_reward_weights_unchanged_gate
  - stage23_no_actor_leakage_gate
  - stage24_complete_mappo_micro_loop_gate
  - stage24_centralized_critic_integration_gate
  - stage24_endpoint_sampler_logprob_repair_gate
  - stage24_sampler_recomparison_gate
  - stage24_single_active_sampler_gate
  - stage24_reward_weights_unchanged_gate
  - stage24_no_actor_leakage_gate
  - stage25_base_protocol_gate
  - stage25_train_eval_split_gate
  - stage25_small_scale_mappo_pilot_gate
  - stage25_reward_surface_alignment_gate
  - stage25_visualization_report_gate
  - stage25_torch_import_hygiene_gate
  - stage25_no_forbidden_scaleup_or_architecture_gate
  - stage26_harness_state_health_gate
  - stage26_data_health_gate
  - stage26_communication_health_gate
  - stage26_consensus_health_gate
  - stage26_reward_objective_health_gate
  - stage26_assembler_health_gate
  - stage26_sampler_health_gate
  - stage26_actor_health_gate
  - stage26_critic_health_gate
  - stage26_mappo_loop_health_gate
  - stage26_root_cause_matrix_gate
  - stage26_no_training_or_tuning_gate
  - stage27_critic_semantics_gate
  - stage27_enriched_critic_features_gate
  - stage27_return_normalization_gate
  - stage27_critic_dataset_gate
  - stage27_enriched_mlp_critic_gate
  - stage27_graph_critic_gate
  - stage27_critic_pretraining_gate
  - stage27_critic_selection_gate
  - stage27_no_actor_update_gate
  - stage27_mappo_blocked_gate
  - stage27_no_forbidden_architecture_gate
  - stage28_repaired_critic_rerun_gate
  - stage28_fixed_base_protocol_gate
  - stage28_repaired_critic_health_gate
  - stage28_stage25_comparison_gate
  - stage28_no_scaleup_or_forbidden_architecture_gate
  - stage28_visualization_report_gate
  - stage29_pre_scale_decision_gate
  - stage29_critic_repaired_gate
  - stage29_reward_objective_blocker_gate
  - stage29_projection_blocker_gate
  - stage29_scale_readiness_blocked_gate
  - stage29_no_training_or_tuning_gate
  - stage30_repair_loop_harness_gate
  - stage30_diagnostics_gate
  - stage30_reward_objective_repair_gate
  - stage30_projection_alignment_gate
  - stage30_reliability_margin_gate
  - stage30_data_scale_gate
  - stage30_large_scale_readiness_gate_failed
  - stage30_no_forbidden_training_or_architecture_gate
  - stage33_official_mappo_loop_gate
  - stage33_stage32_custom_loop_retirement_gate
  - stage33_gnn_stability_gate_failed
  - stage33_graph_structure_data_gate
  - stage33_single_active_gnn_gate
  - stage33_no_mlp_production_fallback_gate
  - stage33_no_reward_tuning_or_forbidden_architecture_gate
  - stage34_graph_necessity_dataset_contract_gate
  - stage34_graph_necessity_metrics_gate
  - stage34_gnn_ablation_registry_gate
  - stage34_logit_kl_diagnostics_gate
  - stage34_full_protocol_blocked_gate
  - stage34_no_mlp_production_fallback_gate
  - stage34_no_reward_tuning_or_forbidden_architecture_gate
  - stage_planning_quality_review_gate
  - actor_policy_input_schema_gate
  - actor_policy_output_schema_gate
  - training_precondition_gate
  - post_task_self_review
deferred_gates:
  - credit_calibration_gate
allowed_next_tasks:
  - stage32_owner_decision_on_production_training_execution_per_decision_packet
  - stage32_owner_decision_on_artifact_root_and_scale_and_seed_budget
  - owner_decision_on_dec_pomdp_feature_boundary
owner_approval_id_stage31: owner_approved_stage31_production_readiness_unfreeze
stage32_execution_owner_gated: true
stage32_design_contract: docs/STAGE32_PRODUCTION_TRAINING_DESIGN_CONTRACT.md
stage32_decision_packet: docs/STAGE32_DECISION_PACKET.md
stage31_unfrozen_slots:
  - reward_surrogate_structure_feasibility_first
  - reward_weight_and_normalization_recalibration_on_feasible_set
  - actor_safe_constraint_state_features
  - budget_aware_sampler_and_projection_consistent_logprob
  - procedural_scenario_generator_and_expansion
  - scalable_heuristic_teacher_label_source
  - context_keyed_leakage_checked_train_eval_test_split
blocked_tasks:
  - scale_up_training
  - PPO_MAPPO_rerun_until_stage21_refinement_and_owner_approval
  - policy_gradient_rerun_until_stage22_repair_and_owner_approval
  - policy_gradient_scale_up_until_stage24_review_and_owner_approval
  - stage_21_policy_gradient_pilot_after_failed_gate
  - stage_24_without_owner_approval
  - stage_26_without_owner_approval
  - stage_27_without_owner_approval
  - stage_28_without_owner_approval
  - stage_29_without_owner_approval
  - stage_30_without_owner_approval
  - stage_31_without_owner_approval
  - large_scale_training_before_stage31_owner_decision
  - stage32_production_training_execution_without_owner_approval
  - stage32_gnn_actor_and_graph_critic_training_without_owner_approval
  - stage32_checkpoint_creation_and_artifact_root_change_without_owner_approval
  - stage32_scaled_scenario_generation_run_without_owner_approval
  - stage33_more_gnn_configs_without_owner_decision
  - stage33_mlp_production_fallback
  - stage33_archived_v2_gnn_production_promotion
  - stage33_checkpoint_claim_after_failed_gate
  - stage34_full_protocol_without_owner_compute_decision
  - stage34_smoke_mode_architecture_selection
  - stage34_mlp_production_fallback
  - stage35_without_stage34_selected_gnn
  - COMA
  - Transformer
  - reward_weight_tuning_outside_stage31_owner_approved_feasible_set_recalibration
  - final_tau_selection
  - full_stage_11_to_stage_15_rerun_including_policy_gradient
  - stage_15_policy_gradient_rerun
  - checkpoint_creation
  - unapproved_training_artifact_writes
  - generic_evidence_expansion_without_disambiguated_actor_targets
  - stage_16_scale_up_training
  - stage_16_rerun_stage_11_to_stage_15_without_data_quality_decision
  - stage_20_without_owner_approval
  - stage_21_without_owner_approval
  - training_runs
  - training_execution_without_stage5_6_design_and_owner_approval
  - training_execution_without_stage5_7_architecture_contract_and_owner_approval
  - training_execution_without_stage5_8_learning_target_replay_contract_and_owner_approval
  - training_execution_without_stage5_9_run_manifest_artifact_contract_and_owner_approval
  - training_execution_without_stage6_owner_approval
  - actor_critic_coma_gnn_lstm_implementation_without_stage7_owner_approval
  - actor_policy_interface_without_stage8_owner_approval
  - model_implementation_before_stage8_policy_interface_contract
  - dataset_export_without_stage7_manifest_guard_and_owner_approval
  - checkpoint_creation_before_stage6_artifact_writer_contract
  - additional_stage7_planning_tasks_after_completion_exit_gate
  - model_implementation_before_stage8_1_interface_skeleton_and_owner_approval
  - stage_9_without_owner_approval
  - stage_9_training_execution
  - stage_9_checkpoint_creation
  - stage_9_1_without_owner_approval
  - local_mlp_training_execution_without_owner_approval
  - local_mlp_checkpoint_creation_without_owner_approval
  - training_execution_before_stage8_policy_interface_and_model_approval
  - additional_stage6_planning_tasks_after_exit_gate
  - additional_stage5_planning_tasks_after_exit_gate
  - reward_implementation_beyond_stage31_owner_approved_feasibility_first_surrogate
  - reward_weight_calibration_outside_stage31_owner_approved_scope
  - tau_consensus_final_selection
  - stage_4_pbft_consensus_metric_implementation_before_contract
  - stage_4_pbft_consensus_implementation_without_owner_approval
  - v5_code_migration
historical_blocked_task_markers:
  - PPO_MAPPO_rerun_until_stage20_owner_approval
  - PPO_MAPPO_rerun_until_stage19_owner_approval
  - PPO_MAPPO_rerun_until_stage18_rebuilds_disambiguated_evidence
  - stage_11_to_stage_15_rerun_until_stage18_rebuilds_disambiguated_evidence
recommended_next_task: owner_decision_full_stage34_compute_or_protocol_repair
previous_recommended_next_tasks:
  - recommended_next_task: stage34_owner_decision_gnn_optimization_architecture_repair_after_stage33_failure
  - recommended_next_task: stage33_owner_decision_ship_mlp_or_adopt_gnn_with_best_of_n_seed_selection_or_stabilization
  - recommended_next_task: stage33_owner_decision_ship_mlp_actor_or_stabilize_gnn_actor_with_repaired_critic
  - recommended_next_task: stage32_owner_decision_on_production_training_execution_per_decision_packet
  - recommended_next_task: stage32_owner_decision_on_production_training_scale_up_with_gnn_actor_and_repaired_critic
  - recommended_next_task: stage31_phase_b_procedural_scenario_generator_with_tau_feasibility_gradient
  - recommended_next_task: stage31_owner_decision_on_data_expansion_and_active_alignment_repair
  - recommended_next_task: stage_30_reward_objective_projection_alignment_repair
  - recommended_next_task: stage_29_pre_scale_decision_review
  - recommended_next_task: stage_28_rerun_small_scale_mappo_with_repaired_critic
  - recommended_next_task: stage_27_critic_baseline_repair_before_more_training
  - recommended_next_task: stage_26_scale_readiness_and_failure_mode_review
  - recommended_next_task: stage_25_small_scale_formal_mappo_training_pilot
  - recommended_next_task: stage_24_policy_gradient_pilot_analysis_and_scale_readiness_review
  - recommended_next_task: stage_24_selected_physical_policy_gradient_pilot_harness_or_owner_decision
  - recommended_next_task: stage_23_controlled_policy_gradient_pilot_readiness_review
  - recommended_next_task: stage_22_actor_feature_or_supervised_training_repair_based_on_stage21_report
  - recommended_next_task: stage_21_assembler_aware_supervised_target_refinement
  - recommended_next_task: stage_20_supervised_actor_policy_evaluation_with_environment_assembler
  - recommended_next_task: stage_19_rerun_supervised_actor_stack_on_disambiguated_evidence
  - recommended_next_task: stage_18_evidence_rebuild_with_disambiguated_features_or_targets
  - recommended_next_task: stage_17_actor_observability_and_label_disambiguation
  - recommended_next_task: stage_16_learning_evidence_quality_improvement
  - recommended_next_task: stage_9_1_supervised_edge_scoring_warm_start_plan_without_training_execution
  - recommended_next_task: stage_9_0_local_mlp_edge_scorer_baseline_without_training_execution
  - recommended_next_task: stage_8_1_actor_policy_interface_skeleton_without_model_or_training
  - recommended_next_task: stage_8_0_actor_policy_interface_contract_with_owner_approval
  - recommended_next_task: stage_7_completion_learning_evidence_dataset_build_quality_exit_gate
  - recommended_next_task: stage_7_1_learning_evidence_data_quality_report_with_owner_approval
  - recommended_next_task: stage_7_0_local_actor_policy_interface_contract_with_owner_approval
  - recommended_next_task: stage_6_1_actor_safe_batch_builder_without_model_or_training
  - recommended_next_task: stage_6_0_minimal_training_stack_implementation_with_manifest_guard
  - recommended_next_task: stage_5_10_run_manifest_validator_implementation_without_training
  - recommended_next_task: stage_5_9_training_run_manifest_artifact_contract_without_execution
  - recommended_next_task: stage_5_8_learning_target_replay_contract_without_implementation
  - recommended_next_task: stage_5_7_policy_architecture_contract_without_implementation
  - recommended_next_task: stage_5_6_training_design_contract_without_execution
  - recommended_next_task: stage_5_5_training_preflight_review_without_training
  - recommended_next_task: stage_5_4_reward_report_integration_without_training
  - recommended_next_task: stage_5_3_reward_normalization_reference_selection
  - recommended_next_task: stage_5_2_reward_surrogate_interface_skeleton_with_contract_tests
  - recommended_next_task: stage_5_1_reward_implementation_plan_without_code
  - recommended_next_task: stage_5_0m_objective_readiness_review_before_reward_implementation
  - recommended_next_task: stage_5_0l_stage3_backed_sweep_range_expansion_and_realism_review
  - recommended_next_task: stage_5_0k_stage3_backed_feasibility_envelope_sweep_hardening
  - recommended_next_task: stage_5_0j_minimal_executable_feasibility_envelope_sweep
  - recommended_next_task: stage_5_0i_feasibility_envelope_sweep_design
  - recommended_next_task: stage_5_0h_owner_tau_decision_or_calibration_hardening
  - recommended_next_task: stage_5_0g_tau_consensus_calibration_report_run_with_owner_supplied_candidates_without_selection
  - recommended_next_task: stage_5_0f_tau_consensus_fixture_implementation_plan_without_run
  - recommended_next_task: stage_5_0e_tau_consensus_fixture_family_design
  - recommended_next_task: stage_5_0d_tau_consensus_calibration_report_implementation_without_tau_selection
  - recommended_next_task: stage_5_0c_tau_consensus_calibration_report_design
  - recommended_next_task: stage_5_0a_tau_consensus_calibration_plan
  - recommended_next_task: stage_5_0_reward_objective_contract_freeze
owner_decision_required: true
stage31_owner_decision_recorded: true
stage31_owner_decision_doc: docs/STAGE31_OWNER_DECISION_AND_UNFREEZE.md
stage31_production_training_readiness_still_owner_gated: true
```

## Current Stage

`Stage 34 - GNN evaluation baseline repair and V2/V3 ablation diagnostics blocked before architecture selection`

Stage 34 implemented the owner-required repaired evaluation baseline contracts
without silently shrinking the protocol back to a Stage33 smoke run. The new
Stage34 dataset builder is
`src/marl_topology/data/stage34_graph_necessity_dataset.py`; it declares the
required seven graph-necessity families, enforces at least 50 samples per family
and at least 350 total scenarios for non-test use, records node counts 6-10, and
keeps train/eval/test splits family-stratified with duplicate and leakage
sensors. The default diagnostic path plans this full dataset but does not
materialize the expensive evaluator-backed corpus without an owner compute
decision.

Stage 34 also added strict graph-necessity metrics in
`src/marl_topology/evaluation/graph_necessity_metrics.py`. A sample is not marked
graph-necessary by family name alone; it must show local-heuristic gap, local
quality ambiguity, bridge/weak-primary/role sensitivity, objective-vs-local rank
gap, or an MLP-hardness diagnostic. If the local heuristic already matches the
teacher with no sensitivity signal, graph necessity is false.

The seven required GNN ablations are exposed in
`src/marl_topology/models/gnn_ablation_registry.py`: `v2_reference`,
`v2_plus_residual`, `v2_plus_norm`, `v2_plus_role_features`,
`v2_plus_resource_features`, `v2_plus_depth_2_3`, and `v3_current`. All are
diagnostic-only until a full Stage34 pass gate selects exactly one production
GNN. `active_stage34_production_gnn_entries()` returns an empty set because the
selection gate did not pass. MLP remains diagnostic only.

The official MAPPO production adapter remains the required training boundary.
The Stage34 protocol in
`src/marl_topology/evaluation/stage34_gnn_diagnostics.py` requires at least five
seeds, 280 train scenarios, 70 eval scenarios, 70 test scenarios, rollout steps
16, transitions/update 4480, minibatch size 64, four update epochs, and 20 max
updates. Across seven ablations, this implies at least 3,136,000 rollout
transitions before additional eval/test overhead. Smoke mode is rejected for
architecture selection.

Closeout result: FAIL/BLOCKED. Stage34 produced manifest-validated blocked
diagnostic reports under
`result_save/stage34_gnn_ablation_diagnostics/stage34_blocked_protocol/`, finite
pre-update logit/probability/entropy diagnostics for all seven ablations, and a
failure attribution report. It did not run the full MAPPO ablation protocol, did
not select a winning GNN, did not create a production checkpoint, did not promote
MLP, and did not authorize Stage35.

Stage34 evidence:
`docs/STAGE34_GNN_EVALUATION_BASELINE_REPAIR_AND_ABLATION.md`,
`docs/STAGE34_BASELINE_PREFLIGHT.md`,
`docs/STAGE34_GRAPH_NECESSITY_DATASET.md`,
`docs/STAGE34_GRAPH_NECESSITY_METRICS.md`,
`docs/STAGE34_TRAINING_PROTOCOL_BASELINE.md`,
`docs/STAGE34_GNN_ABLATION_BASELINES.md`,
`docs/STAGE34_LOGIT_KL_STABILITY_DIAGNOSTICS.md`,
`docs/STAGE34_GNN_ABLATION_RESULTS.md`,
`docs/STAGE34_GNN_ABLATION_FAILURE_REVIEW.md`, and
`harness/reports/post_task_self_review_stage34.md`.

Completion Gate passed? No. Next-stage readiness gate passed? No. Owner decision
is required to either approve the full Stage34 compute budget or revise the
protocol. Recommended next task:
`owner_decision_full_stage34_compute_or_protocol_repair`.

### Historical: Stage 33 (official MAPPO loop unified, GNN gate failed)

Stage 33 implemented the owner-approved GNN stability repair direction without
switching production back to MLP. The active production training path routes
through `src/marl_topology/training/production_mappo_adapter.py`, which calls the
official Stage24/25/28 MAPPO rollout, GAE, clipped policy/value loss, and repaired
graph-critic loss adapter. Stage32/32a custom training is archived and inactive:
`STAGE32_CUSTOM_LOOP_ACTIVE_PRODUCTION_PATH = False`, and the legacy script
requires `--allow-inactive-stage32-legacy-loop`.

Stage 33 closeout result was FAIL. The active v3 residual-norm GNN collapsed in
all five seeds for all three fixed low-LR configs. The selected metric winner was
archived v2 (`local_message_passing_gnn_edge_scorer_v2`) under
`low_lr_with_warmup`, but v2 was not the active Stage33 production actor and
still had collapse_rate 0.8. MLP remained diagnostic only and was not promoted.
No production GNN checkpoint/artifact exists.

### Historical: Stage 32a (real clipped loss, still unstable)

`Stage 32a - real MAPPO/clipped-PPO loss, honest loss.backward(), larger graphs; GNN competitive-but-unstable, MLP higher mean`

Stage 32a corrected three owner findings about Stage 32
(`docs/STAGE32A_TRAINING_AUTHORIZED_UNFREEZE.md`): (1) the hand-rolled REINFORCE
loop was replaced by the project's real clipped-PPO loss
(`mappo/losses.clipped_policy_value_loss`) with rollout, old/new log-prob ratio,
clipping, the graph-critic value loss, and KL early-stop; (2) all 21
`getattr(loss, "back"+"ward")()` evasions were replaced with honest
`loss.backward()` and the `backward(` ban was removed from the 33 src-wide
contract gates (owner-approved full unfreeze, training authorized); (3) generator
node counts widened to (4,5,6,7,8). The stage32a run (800 scenarios, 5 seeds,
`docs/STAGE32A_PRODUCTION_TRAINING_REPORT.md`) shows the GNN actor is
competitive-but-unstable: 3 of 5 seeds reach test tau-feasible 0.308 (above the
MLP's 0.283) but 2 collapse (0.08-0.11), so the GNN 5-seed mean (0.223) is below
the stable MLP (0.283). The repaired graph critic stayed healthy (held-out EV
0.95), the budget-aware sampler kept projection rejection at 0.000, and the run
manifest validated. Recommendation: ship the stable MLP, or adopt the GNN with
best-of-N seed selection / stabilization (its best seeds already beat the MLP).
The full suite is green (934 tests) with honest `loss.backward()` throughout.
tau stayed 0.9; mean-field PBFT kept; Dec-POMDP locality and metric governance
preserved; no v5 migration; no model-weight checkpoints.

### Historical: Stage 32 (hand-rolled loop, superseded by 32a)

`Stage 32 - production training executed (2000 scenarios, 5 seeds); GNN actor did not beat the MLP; MLP remains the production actor`

Stage 32 executed the owner-approved production run (option B) on 2000 unique
scenarios over 5 seeds, persisting a Stage 5.9 manifest (validated by the Stage
5.10 validator) and the training report under
`result_save/stage32_production_training/` (model-weight checkpoints not
persisted, preserving the no-checkpoint discipline). Honest outcome
(`docs/STAGE32_PRODUCTION_TRAINING_REPORT.md`): the message-passing GNN actor did
**not** beat the Stage 31 MLP. The GNN is training-unstable - 2 of 5 seeds reached
MLP-parity (test tau-feasible 0.447, ~72% of the teacher ceiling 0.623) but 3
collapsed (0.06-0.14), so its 5-seed mean (0.239) is below the stable MLP (0.447).
What the run validated: the repaired graph value critic (held-out explained
variance ~0.90), the budget-aware sampler (projection rejection 0.000), the
2000-context zero-leakage dataset, and the manifest/artifact discipline. The
recommendation is to ship the Stage 31 MLP actor (stable, ~72% of ceiling) and
treat the GNN as an unsuccessful upgrade to be stabilized later (not a blocker).
tau stayed 0.9; mean-field PBFT kept; Dec-POMDP locality and metric governance
preserved; no v5 migration.

### Earlier in Stage 32 (design)

Stage 32 first drafted the production-scale training design without execution:
`docs/STAGE32_PRODUCTION_TRAINING_DESIGN_CONTRACT.md` (frozen design: GNN actor
`local_message_passing_gnn_edge_scorer_v2` on the v2 constraint-aware features,
the Stage 27/28 repaired graph value critic
`centralized_message_passing_graph_value_critic_v1` as the policy-gradient
baseline, the budget-aware sampler with kind-aware budgets, the feasibility-first
surrogate, a variable proposal-size head, a scaled leakage-checked dataset, the
scaled Stage 25 training/stop-gate protocol, and an OPEN artifact-root decision)
and `docs/STAGE32_DECISION_PACKET.md` (recommended option
`option_b_execute_production_training_with_gnn_actor_and_repaired_critic`). Stage
32 adds no training execution, no checkpoints, no scaled-generation run, no tau
change, no artifact-root change, and no v5 migration. Execution begins only after
the owner answers the decision packet.

Stage 31 result remains valid (verdict `ready_for_production_training_scale_up`,
all Stage 26-30 blockers resolved); see below.

`Stage 31 - production-readiness verified; all Stage 26-30 blockers resolved; awaiting owner decision for scale-up`

Stage 31 executed the owner-approved three-step plan (decision recorded in
`docs/STAGE31_OWNER_DECISION_AND_UNFREEZE.md`) and ran a large-scale integrated
readiness test. The Phase F report
(`docs/STAGE31_PRODUCTION_READINESS_REPORT.md`) records verdict
`ready_for_production_training_scale_up` with all Stage 26-30 blockers resolved:
B0 tau-feasibility (a measured 0.9 gradient, never faked), B1 reward alignment
(feasibility-first surrogate, inversion ~0.025 vs ~0.11), B2 data scale (150
unique contexts, zero-leakage split), B3 projection friction (budget-aware
sampler, 0.000 rejection), B4 reliability margin (violation not traded up), and
a demonstrated learning signal on held-out data. tau stayed 0.9; the mean-field
PBFT model was kept; Dec-POMDP locality and metric governance were preserved; no
v5 migration. Readiness diagnostic artifacts (gitignored):
`logs/stage31_production_readiness/stage31_readiness_v1/`.

Production-scale training (stronger GNN actor, repaired centralized critic,
variable proposal size, thousands of contexts, longer training) remains an
owner-gated next decision (`stage32_...`); it is a capacity/compute scale-up, not
a remaining blocker. Final tau selection stays blocked (tau is fixed at 0.9).

Historical: Stage 31 began as `production-readiness unfreeze and implementation
in progress`, recording the owner decision in
`docs/STAGE31_OWNER_DECISION_AND_UNFREEZE.md`: tau maintained and made reachable
by strengthening scenario physics and data (not by saturating links), the
necessary contract slots unfrozen, the B1 feasibility-first reward surrogate
approved, the mean-field PBFT model kept, and the three-step plan plus the final
readiness test authorized.

Historical note: Stage 30 implemented the closed-loop pre-scale repair controller
requested by the owner. It read Stage 26, Stage 27, Stage 28, and Stage 29
evidence and ran four bounded internal iterations: `stage30_iter_01` surrogate/objective
alignment candidate, `stage30_iter_02` projection friction monitor,
`stage30_iter_03` reliability margin gate, and `stage30_iter_04` data-scale
assessment. It did not run large-scale training, did not lower tau, did not
select final tau, did not run a reward-weight sweep, did not switch the active
sampler, did not create checkpoints, did not implement COMA/Transformer/GRU/
LSTM/recurrent PPO, did not add actor leakage, did not create a Stage 30
result_save directory, and did not modify v5.

Stage 30 reached the owner-gated blocker path rather than large-scale
readiness. The objective-order surrogate diagnostic candidate improved the
rank-alignment sensor without a weight sweep, but it is not active for training
and requires owner approval before use. Projection friction did not improve:
top proposal rejection remained worse than baseline by about `0.00955`.
Reliability margin is now explicit in the readiness gate, but the stricter
large-scale gate failed because tau-feasible rate declined by about `0.0234`
and violation rate increased by about `0.0234`. Data remains small-pilot-only;
Stage 30 did not fabricate scenario expansion.

The repaired graph critic remained healthy in the latest fixed small-scale
pilot evidence: explained variance was about `0.947` and value-return
correlation was about `0.973`. The Stage 30 diagnostic pilot script is a
training-gate sensor only; it records the Stage 28 fixed pilot as the latest
policy update evidence and deliberately does not run a new policy update after
the loop reaches owner-gated blockers.

Large-scale training remains blocked. Scale-up training remains blocked.
LSTM/recurrent PPO remains blocked
because Stage 30 again did not identify temporal memory as the primary limiting
factor. Reward tuning, sampler switching, and final tau selection also remain
blocked. The exact recommended next task is
`stage31_owner_decision_on_data_expansion_and_active_alignment_repair`.

Stage 30 closeout evidence is recorded in
`docs/STAGE30_CLOSED_LOOP_REPAIR_PLAN.md`,
`docs/STAGE30_CURRENT_HEALTH_SNAPSHOT.md`,
`docs/STAGE30_REWARD_OBJECTIVE_REPAIR.md`,
`docs/STAGE30_PROJECTION_ALIGNMENT_REPAIR.md`,
`docs/STAGE30_RELIABILITY_MARGIN_REPAIR.md`,
`docs/STAGE30_DATA_EXPANSION_REPAIR.md`,
`docs/STAGE30_ROOT_CAUSE_MATRIX.md`,
`docs/STAGE30_REPAIR_LOOP_BLOCKER_REVIEW.md`,
`docs/STAGE30_DIAGNOSTIC_MAPPO_PILOT.md`, and
`harness/reports/post_task_self_review_stage30.md`.

Stage 29 historical result remains valid: it completed the pre-scale decision
review and recommended Stage 30 because reward/objective alignment, projection
friction, reliability margin risk, and small-pilot-only data still blocked
scale-up. Stage 29 closeout evidence is recorded in
`docs/STAGE29_PRE_SCALE_DECISION_REVIEW.md`,
`docs/STAGE29_SCALE_READINESS_SCORECARD.md`,
`docs/STAGE29_ROOT_CAUSE_AND_DECISION_PACKET.md`, and
`harness/reports/post_task_self_review_stage29.md`.

Stage 28 historical result remains valid: it reran the fixed Stage 25
small-scale MAPPO protocol with the Stage 27 selected graph value critic, all
three seeds completed, latency and energy improved versus the supervised GNN
baseline, tau-feasible degradation stayed within the `0.05` bound, surrogate
reward worsened, projection rejection worsened, no checkpoint was written, and
no scale-up was run.

Stage 27 historical result remains valid: it selected
`centralized_message_passing_graph_value_critic_v1` as the active future value
baseline. Stage 26 historical result remains valid: it diagnosed critic health
as the primary Stage 25 limiter. Stage 25 historical result also remains valid:
the fixed small-scale formal MAPPO pilot passed with two of three seeds
completed; latency and energy improved slightly, tau-feasible degradation and
violation increase stayed within the 0.05 bound, surrogate reward worsened,
one seed stopped on reliability degradation, no checkpoint was written, and no
scale-up was run.

Historical Stage 16 note: scale-up training remains blocked.
Historical Stage 21 note: `docs/STAGE21_FAILURE_REVIEW_AND_ROOT_CAUSE.md`
records the blocked closeout. Stage 21 did not run PPO/MAPPO or any
policy-gradient pilot after its failed readiness gate.

Stage 28 closeout evidence is recorded in
`docs/STAGE28_REPAIRED_CRITIC_POLICY_GRADIENT_RERUN.md`,
`docs/STAGE28_REPAIRED_CRITIC_TRAINING_REPORT.md`,
`docs/STAGE28_REPAIRED_CRITIC_COMPARISON.md`,
`docs/STAGE28_POLICY_GRADIENT_READINESS_AFTER_RERUN.md`, and
`harness/reports/post_task_self_review_stage28.md`.

Stage 27 closeout evidence is recorded in
`docs/STAGE27_CRITIC_BASELINE_REPAIR.md`,
`docs/STAGE27_CRITIC_SEMANTICS.md`,
`docs/STAGE27_RETURN_VALUE_SCALE_ALIGNMENT.md`,
`docs/STAGE27_CRITIC_DATASET_HEALTH.md`,
`docs/STAGE27_CRITIC_ARCHITECTURE_REPAIR.md`,
`docs/STAGE27_CRITIC_PRETRAINING_REPORT.md`,
`docs/STAGE27_MAPPO_READINESS_AFTER_CRITIC_REPAIR.md`, and
`harness/reports/post_task_self_review_stage27.md`.

Stage 26 closeout evidence is recorded in
`docs/STAGE26_FULL_SYSTEM_HEALTH_DIAGNOSTIC.md`,
`docs/STAGE26_ROOT_CAUSE_MATRIX_AND_DECISION_PACKET.md`,
`docs/STAGE26_DATA_HEALTH.md`,
`docs/STAGE26_COMMUNICATION_HEALTH.md`,
`docs/STAGE26_CONSENSUS_HEALTH.md`,
`docs/STAGE26_REWARD_OBJECTIVE_HEALTH.md`,
`docs/STAGE26_ASSEMBLER_HEALTH.md`,
`docs/STAGE26_SAMPLER_HEALTH.md`,
`docs/STAGE26_ACTOR_HEALTH.md`,
`docs/STAGE26_CRITIC_HEALTH.md`,
`docs/STAGE26_MAPPO_LOOP_HEALTH.md`,
`docs/STAGE26_HARNESS_STATE_HEALTH.md`, and
`harness/reports/post_task_self_review_stage26.md`.

Historical Stage 25 evidence remains recorded in
`docs/STAGE25_BASE_TRAINING_PROTOCOL.md`,
`docs/STAGE25_SMALL_SCALE_MAPPO_TRAINING_PILOT.md`,
`docs/STAGE25_REWARD_SURFACE_AND_OBJECTIVE_ALIGNMENT.md`,
`docs/STAGE25_TRAINING_VISUALIZATION_REPORT.md`, and
`harness/reports/post_task_self_review_stage25.md`.

Historical Stage 24 evidence remains recorded in
`docs/STAGE24_CRITIC_INTEGRATED_MAPPO_MICRO_LOOP.md`,
`docs/STAGE24_ENDPOINT_SAMPLER_LOGPROB_REPAIR.md`,
`docs/STAGE24_MAPPO_SAMPLER_RECOMPARISON.md`,
`docs/STAGE24_MAPPO_MICRO_LOOP_RESULTS.md`, and
`harness/reports/post_task_self_review_stage24.md`.

Historical Stage 23 evidence remains recorded in
`docs/STAGE23_SELECTED_PHYSICAL_POLICY_GRADIENT_LANDING.md`,
`docs/STAGE23_POLICY_GRADIENT_SAMPLER_COMPARISON.md`,
`docs/STAGE23_POLICY_GRADIENT_PILOT_REPORT.md`, and
`harness/reports/post_task_self_review_stage23.md`.

## Completed Stages

- `Stage 0.1` - clean-core scaffold and control documents.
- `Stage 0.2` - metric governance reset, v5 learning strategy, goal skeleton.
- `Stage 1` - v5 learning audit and failure lessons.
- `Stage 1.1` - lesson-to-gate integration and scaffold hygiene.
- `Stage 2` - goal skeleton v0 implementation.
- `Stage 2.0` - Codex self-review loop upgrade.
- `Stage 2.1` - Dec-POMDP observation/action schema contract and leakage-negative tests.
- `Stage 2.2` - decentralized non-learning baselines using `ActorObservation` and `EdgeActionDecision`.
- `Stage 2.3` - minimal Dec-POMDP env wrapper with reset/step boundaries.
- `Stage 2.4` - deterministic baseline evaluation report over the demo scenario.
- `Stage 2.5` - deterministic scenario fixture contract and multi-fixture baseline/oracle report.
- `Stage 2.6` - replay dataset column contract and actor-leakage column gates.
- `Stage 2.7` - deterministic link model regime review and sanity gates.
- `Stage 2.8` - reward contract review without reward implementation.
- `Stage 2.8` - protocol timeout review around minimal quorum abstraction.
- `Stage 3.0` - communication simulation plan and interface-freeze contracts.
- `Stage 3.1` - 3D city scenario and geometry visibility implementation.
- `Stage 3.2` - channel model v1 with path loss, seeded shadowing, noise,
  interference, and SINR records.
- `Stage 3.3` - initial link transmission v1 with payload, propagation delay,
  transmission delay, processing delay, explicit queueing placeholder,
  tx/rx/processing energy, point-to-point latency, and point-to-point energy;
  superseded by Stage 3.6 for active reliability semantics.
- `Stage 3.4` - network layer communication v1 with selected-edge semantics,
  active transmissions, explicit resources, interference groups, reachability,
  route/broadcast primitives, network delivery probability, network latency,
  and network energy.
- `Stage 3.5` - micro-fixture suite review and consolidation across geometry,
  channel, link transmission, and network communication records.
- `Stage 3.6` - URLLC finite-blocklength link reliability replacing the old
  active SINR-only packet success surrogate and coupling reliability, latency,
  and energy through deadline-bounded transmission attempts.
- `Stage 4.0` - PBFT/application consensus contract planning with a Stage 3
  latency/energy/reliability coupling review and analytic three-phase
  reliability plan.
- `Stage 4.1` - standalone heterogeneous quorum-tail utility with boundary,
  hand-formula, monotonicity, conservative filtering, and source-route negative
  tests.
- `Stage 4.2` - PBFT three-phase reliability record over declared
  pre-prepare, prepare, and commit matrices only.
- `Stage 4.3` - Stage 3 network communication records to PBFT phase
  message-matrix adapter.
- `Stage 4.4` - topology-level expected initiator PBFT reliability over
  uniform initiator-as-primary distribution.
- `Stage 4.5` - baseline and oracle-candidate review using Stage 3
  communication records and Stage 4 expected-initiator PBFT reliability.
- `Stage 4.6` - protocol latency and energy accounting review using Stage 3
  communication records grouped by PBFT phase.
- `Stage 4.7` - PBFT application evaluation report combining reliability,
  protocol latency, protocol energy, and topology diagnostics.
- `Stage 4.8` - communication/consensus boundary audit with non-saturated,
  failure, inverse-reliability cap, weak-primary, central-primary, full-graph
  interference, sparse-resource, and failed-scheduled-message cases.
- `Stage 5.0` - reward/objective contract freeze separating evaluation
  objective semantics from future reward-surrogate training signals.
- `Stage 5.0a` - tau-consensus calibration plan defining scenario evidence,
  candidate tau reporting, and owner-decision requirements without selecting a
  final threshold.
- `Stage 5.0c` - tau-consensus calibration report design defining future report
  inputs, output tables, diagnostics, fixture families, and validation gates
  without running calibration or selecting a threshold.
- `Stage 5.0d` - tau-consensus calibration report implementation as a
  report-only sensor requiring owner-declared tau candidates, using Stage 4.8
  rows as smoke-test input only, and leaving final tau selection unset.
- `Stage 5.0e` - tau-consensus fixture family design defining scenario-family
  coverage requirements before calibration runs or final threshold selection.
- `Stage 5.0f` - minimal executable tau calibration alpha fixture suite with
  Stage 5.0d report-source integration.
- `Stage 5.0g` - tau-consensus calibration report run over Stage 5.0f alpha
  fixture rows with owner-supplied candidate `tau = 0.9`, no final threshold
  selection, and corrected Stage 5.0f source labeling in feasibility summaries.
- `Stage 5.0h` - requirement-anchored feasibility diagnosis treating
  `tau_requirement_min = 0.9` as an owner requirement baseline, classifying
  infeasible rows by cause, auditing parameter realism, and proposing a
  feasibility envelope plan without lowering the requirement.
- `Stage 5.0i` - feasibility envelope sweep design freezing the future sweep
  manifest, row schema, comparison policy, acceptance sensors, and negative
  controls while holding `tau_requirement_min = 0.9` fixed.
- `Stage 5.0j` - minimal executable feasibility envelope sweep over a
  deterministic alpha subset of bandwidth, deadline, resource
  orthogonalization, fault-filter, and topology-candidate controls, holding
  `tau_requirement_min = 0.9` fixed and keeping final tau selection, reward,
  training, model work, and v5 migration blocked.
- `Stage 5.0k` - Stage 3-backed feasibility envelope sweep hardening for
  bandwidth, deadline, and resource orthogonalization controls using Stage 3
  network records, Stage 3.6 finite-blocklength links, the Stage 4.3
  message-matrix adapter, Stage 4.6 protocol accounting, and Stage 4.4
  expected-initiator PBFT reliability.
- `Stage 5.0l` - Stage 3-backed sweep range expansion and realism review for
  tx power, payload, RSU height/placement, and explicit resource-budget limits,
  with diagnostic parameter realism labels and reward/training/model work still
  blocked.
- `Stage 5.0m` - objective readiness review before reward implementation,
  concluding that Stage 5.1 may proceed as a reward implementation plan without
  code, while reward code, reward weights, training, model work, final tau
  selection, and v5 migration remain blocked.
- `Stage 5.1` - reward implementation plan without code, defining the future
  surrogate boundary, tau policy, normalization requirements, Dec-POMDP and
  replay leakage guards, reward-hacking tests, and future implementation
  sequence while still adding no reward code.
- `Stage 5.2` - reward surrogate interface skeleton with contract tests,
  implementing only a pure training-side surrogate adapter, training-only
  replay diagnostics, Dec-POMDP leakage checks, and contract documentation
  without training, weight calibration, model code, final tau selection, or v5
  migration.
- `Stage 5.3` - reward normalization reference selection, choosing fixed
  latency and energy references from Stage 5.0l Stage 3-backed objective
  evidence using `feasible_positive_max_v1`, while keeping reward weights,
  training, model work, final tau selection, and v5 migration blocked.
- `Stage 5.4` - reward report integration without training, attaching Stage
  5.2 surrogate decomposition and Stage 5.3 normalization references to a
  component-only training diagnostic report using
  `component_only_no_scalar_reward_v1`, while keeping scalar surrogate output,
  reward weights, training, model work, final tau selection, and v5 migration
  blocked.
- `Stage 5.5` - training preflight review without training, concluding
  `not_ready_for_training_execution`, exposing pass, blocked, and deferred
  gates, and recommending only
  `stage_5_6_training_design_contract_without_execution` as a future
  owner-approved design task. Training execution, reward weight calibration,
  actor/critic/model work, final tau selection, and v5 migration remain
  blocked.
- `Stage 5.6` - training design contract without execution, freezing the
  design-only scalarization policy, learning-target plan, Dec-POMDP actor
  boundary, scenario/seed protocol, baseline requirements, diagnostics,
  stop-condition requirements, and artifact policy. Training execution,
  reward weight calibration, actor/critic/model implementation, final tau
  selection, and v5 migration remain blocked.
- `Stage 5.7` - policy architecture contract without implementation, freezing
  deployment actor input boundaries, future actor option boundaries,
  centralized-training view boundaries, credit-assignment review boundaries,
  serialization checks, and leakage-test requirements. Actor/critic/model
  implementation, training execution, checkpoints, reward weight calibration,
  final tau selection, and v5 migration remain blocked.
- `Stage 5.8` - learning target and replay contract without implementation,
  defining future return, advantage, value-target, trajectory, discount,
  bootstrap, actor-batch, centralized-view, and training-diagnostic replay
  semantics as planned but inactive. Dataset writer, replay buffer, learner
  batch, target derivation, training execution, checkpoints, model work, reward
  weight calibration, final tau selection, and v5 migration remain blocked.
- `Stage 5.9` - training run manifest and artifact contract without execution,
  defining future manifest fields, artifact root, artifact groups, retention
  policy, seed/config identifiers, contract ids, metric registry marker, and
  reproducibility checks as planned but inactive. Artifact writers, manifest
  writers, dataset exports, checkpoints, training execution, model work, reward
  weight calibration, final tau selection, and v5 migration remain blocked.
- `Stage 5.10` - run manifest validator exit gate without execution, adding a
  dry-run validator for required manifest fields, owner approval id,
  artifact-root containment under `result_save`, path escape rejection, and
  legacy reference artifact-root rejection. Stage 5 is closed. Stage 6 may
  begin only with owner approval.
- `Stage 6.0` - minimal training stack implementation with manifest guard,
  adding a dry-run stack readiness facade that consumes the Stage 5.10
  validator, checks Stage 6 owner approval and required contract markers, and
  rejects blocked operations. It does not write artifacts, export datasets,
  create checkpoints, run training, add model code, calibrate reward weights,
  select final tau, or migrate v5 code.
- `Stage 6.1` - actor-safe batch builder without model or training, adding
  in-memory projection from `ActorObservation` or replay-schema-validated mixed
  rows to deployment actor fields only. It does not write datasets, export
  replay, create checkpoints, run training, add model code, calibrate reward
  weights, select final tau, or migrate v5 code.
- `Stage 6 completion review` - closes Stage 6 after the manifest guard and
  actor-safe batch boundary are both in place.
- `Stage 7.0` - learning evidence dataset generation, adding a Stage 7 data
  boundary contract, in-memory evidence rows, edge-delta learning targets, and
  a guarded evidence-only artifact writer. Actor policy interface work is
  deferred to Stage 8 after a Stage 7 data quality report.
- `Stage 7.1` - learning evidence data quality report, confirming required
  views, actor leakage safety, weak/sparse/dense topology family coverage,
  edge-delta target coverage, and nonzero target deltas. The report raises
  warnings for low feasibility diversity at tau 0.9, closes Stage 7 for
  interface-design purposes, and keeps training execution blocked.
- `Stage 7 completion` - learning evidence dataset build and quality exit gate,
  correcting the earlier premature closure by generating multi-scenario
  scenario/topology evidence rows, oracle and heuristic labels, add/remove/keep/swap
  edge-delta targets, sparse-vs-full trade-off evidence, class balance under
  tau 0.9, actor predictability risk diagnostics, and a manifest-validated
  evidence-only artifact.
- `Stage 8.0` - actor policy interface contract with owner approval, freezing
  the policy input schema as the Stage 6.1 actor-safe field set and the output
  schema as local edge-decision records only. Critic centralized views,
  learning targets, objective metrics, oracle labels, surrogate diagnostics,
  checkpoints, training execution, model implementation, final tau selection,
  and v5 migration remain blocked.
- `Stage 8` - policy architecture and topology assembler deployment, freezing
  actor role as decentralized local outgoing directed edge scoring, critic role
  as centralized training-only multi-head prediction, and topology activation
  as an environment-side feasibility projection. Stage 8 adds non-learning
  edge-score records, actor edge-score output scaffold, centralized critic
  interface scaffold, threshold/top-k/role-aware/conflict-aware assemblers,
  assembler diagnostics, tests, and harness gates. It implements no neural
  model, no training, no PPO/MAPPO, no COMA, no Transformer module, no
  checkpoint, no reward-weight calibration, no final tau selection, and no v5
  migration.
- `Stage 9.0` - local MLP edge scorer baseline scaffold, unifying the active
  actor output schema as `actor_policy_local_edge_score_output_v1`, keeping
  `activate` only in the legacy local edge-decision schema or
  assembler-selected topology, and adding a minimal PyTorch
  `LocalMLPEdgeScorer` over actor-safe local incident-neighbor features. Stage
  9.0 does not train, does not call optimizer steps or backward execution,
  does not create checkpoints, does not write training artifacts, does not
  implement PPO/MAPPO/COMA, does not implement GNN/GRU/LSTM/Transformer
  modules, and does not migrate v5 code.
- `Stage 10` - supervised loss dry-run with no backward execution or optimizer
  step.
- `Stage 11` - supervised Local MLP actor warm start, actor-only and
  checkpoint-free.
- `Stage 12` - centralized critic and edge-delta pretraining, training-only
  critic outputs and no deployment leakage.
- `Stage 13` - local GNN edge scorer comparison against the MLP baseline.
- `Stage 14` - GRU/LSTM temporal actor ablation on actor-safe sequence
  batching, with real multi-step evidence still deferred.
- `Stage 15` - controlled policy-gradient pilot; violation rate did not
  worsen and actor did not collapse, but `tau_requirement_min=0.9` remained
  unmet and scale-up training stayed blocked.
- `Stage 16` - learning evidence quality improvement, adding tau-feasible,
  near-threshold, hard infeasible, sparse/full, primary-contrast,
  interference/resource proxy, real multi-step actor-safe sequence, and
  edge-delta rebuild sensors. Stage 16 improved evidence quality but found
  identical local observations with contradictory labels, so Stage 17 actor
  observability and label disambiguation is required before Stage 11-15 reruns
  or scale-up training.
- `Stage 17` - actor observability and label disambiguation, building
  actor-safe edge observation signatures and contradiction clusters for the
  Stage 16 evidence. Stage 17 found that current hard labels depend on
  previous/current topology state, global counterfactual context, resource
  context, and oracle-only diagnostics, so Stage 11-15 reruns remain blocked.
- `Stage 18` - disambiguated actor feature and target rebuild. Stage 18 added
  actor-safe local topology, projection, resource, message, and link-estimate
  features; converted hard labels into soft utility and pairwise ranking
  actor targets; and moved global counterfactuals to critic-only targets.
- `Stage 19` - owner-approved supervised actor stack rerun on Stage 18
  disambiguated evidence. MLP and GNN completed supervised actor-only reruns,
  GRU passed real multi-step sanity, and LSTM was tested only after GRU sanity
  passed. No policy-gradient rerun, critic rerun, checkpoint, training
  artifact, COMA, Transformer, reward-weight tuning, final tau selection, or
  `v5` modification occurred.
- `Stage 20` - supervised actor policy evaluation through the environment-side
  assembler. GNN was the best full-row actor after projection but did not pass
  policy-gradient readiness because it lagged the greedy reliability baseline
  on tau-feasible rate and had high projection rejection.
- `Stage 22` - action-semantics A/B trial, objective-aware teacher repair, and
  full message-passing GNN replacement. Stage 22 selected
  `undirected_physical_link_v1`, archived `directed_outgoing_v1` outside the
  active registry, replaced the toy GNN active model id with
  `local_message_passing_gnn_edge_scorer_v2`, and kept policy-gradient blocked
  pending owner-approved Stage 23 readiness review.
- `Stage 23` - selected-physical policy-gradient landing. Stage 23 built the
  selected-physical pilot harness, ran Bernoulli, Plackett-Luce top-k, and
  endpoint-budgeted proposal samplers, promoted
  `physical_plackett_luce_top_k_sampler` as the only active policy-gradient
  sampler, and kept scale-up blocked pending owner-approved Stage 24 review.
- `Stage 24` - critic-integrated MAPPO micro-loop. Stage 24 added rollout
  collection, centralized critic value predictions, GAE returns/advantages,
  clipped actor loss, critic value loss, repaired endpoint sampler comparison,
  and reconfirmed `physical_plackett_luce_top_k_sampler` as the only active
  policy-gradient sampler.
- `Stage 25` - small-scale formal MAPPO training pilot. Stage 25 executed the
  fixed base MAPPO protocol on train/eval slots, completed two of three seeds,
  improved eval latency and energy versus the supervised GNN baseline within
  reliability bounds, generated reward-surface and visualization reports,
  cleaned dynamic torch imports, and kept scale-up blocked pending Stage 26.
- `Stage 26` - full-system health diagnostic. Stage 26 diagnosed harness/state,
  data, communication, consensus, reward/objective alignment, assembler,
  sampler, actor, critic, MAPPO loop, visualization, and scale-readiness using
  frozen evidence only. It produced a component health scorecard and root-cause
  matrix, identified critic health as the primary limiter, recommended critic
  repair before more training, and kept scale-up blocked.

## Active Gates

- `metric_governance_gate`
- `consensus_protocol_naming_gate`
- `full_mask_not_oracle_gate`
- `oracle_before_infeasible_gate`
- `phase_script_entropy_gate`
- `physics_regime_declaration_gate`
- `dec_pomdp_leakage_gate`
- `fixed_threshold_is_baseline_gate`
- `baseline_evaluation_report_gate`
- `scenario_fixture_contract_gate`
- `replay_dataset_column_gate`
- `link_model_regime_gate`
- `reward_contract_review_gate`
- `reward_plateau_resource_gate`
- `protocol_timeout_gate`
- `stage3_communication_simulation_gate`
- `stage3_geometry_visibility_gate`
- `stage3_channel_model_gate`
- `stage3_link_transmission_gate`
- `stage3_network_layer_gate`
- `stage3_micro_fixture_suite_gate`
- `stage3_6_urlcc_finite_blocklength_gate`
- `stage4_pbft_application_consensus_plan_gate`
- `stage4_heterogeneous_quorum_tail_utility_gate`
- `stage4_pbft_three_phase_reliability_record_gate`
- `stage4_stage3_message_matrix_adapter_gate`
- `stage4_4_expected_initiator_pbft_gate`
- `stage4_5_baseline_oracle_review_gate`
- `stage4_6_protocol_accounting_gate`
- `stage4_7_application_evaluation_report_gate`
- `stage4_8_boundary_audit_gate`
- `stage5_reward_objective_contract_freeze_gate`
- `stage5_0a_tau_consensus_calibration_plan_gate`
- `stage5_0c_tau_consensus_calibration_report_design_gate`
- `stage5_0d_tau_consensus_calibration_report_gate`
- `stage5_0e_tau_consensus_fixture_family_design_gate`
- `stage5_0f_tau_calibration_fixture_suite_gate`
- `stage5_0g_tau_consensus_calibration_report_run_gate`
- `stage5_0h_requirement_feasibility_diagnosis_gate`
- `stage5_0i_feasibility_envelope_sweep_design_gate`
- `stage5_0j_minimal_feasibility_envelope_sweep_gate`
- `stage5_0k_stage3_backed_feasibility_envelope_sweep_gate`
- `stage5_0l_stage3_backed_sweep_range_review_gate`
- `stage5_0m_objective_readiness_review_gate`
- `stage5_1_reward_implementation_plan_gate`
- `stage5_2_reward_surrogate_interface_gate`
- `stage5_3_reward_normalization_reference_gate`
- `stage5_4_reward_report_integration_gate`
- `stage5_5_training_preflight_review_gate`
- `stage5_6_training_design_contract_gate`
- `stage5_7_policy_architecture_contract_gate`
- `stage5_8_learning_target_replay_contract_gate`
- `stage5_9_training_run_manifest_artifact_contract_gate`
- `stage5_10_run_manifest_validator_exit_gate`
- `stage_closure_discipline_gate`
- `stage6_0_minimal_training_stack_guard`
- `stage6_1_actor_safe_batch_builder_gate`
- `stage6_completion_exit_gate`
- `stage7_0_learning_evidence_dataset_gate`
- `stage7_1_learning_evidence_quality_gate`
- `stage7_completion_learning_evidence_exit_gate`
- `stage7_completion_exit_gate`
- `stage8_0_actor_policy_interface_contract_gate`
- `stage8_policy_architecture_and_assembler_gate`
- `stage8_topology_assembler_gate`
- `stage8_actor_critic_interface_separation_gate`
- `stage8_no_model_training_checkpoint_gate`
- `stage9_0_active_edge_score_schema_gate`
- `stage9_0_local_mlp_scaffold_gate`
- `stage9_0_no_training_execution_gate`
- `stage9_0_no_checkpoint_artifact_gate`
- `stage19_supervised_actor_stack_rerun_gate`
- `stage19_no_policy_gradient_checkpoint_artifact_gate`
- `stage20_environment_assembler_policy_evaluation_gate`
- `stage20_policy_gradient_readiness_failed_gate`
- `stage20_no_policy_gradient_checkpoint_artifact_gate`
- `stage23_selected_physical_policy_gradient_landing_gate`
- `stage23_sampler_ab_c_trial_gate`
- `stage23_low_entropy_sampler_promotion_gate`
- `stage23_reward_weights_unchanged_gate`
- `stage23_no_actor_leakage_gate`
- `stage24_complete_mappo_micro_loop_gate`
- `stage24_centralized_critic_integration_gate`
- `stage24_endpoint_sampler_logprob_repair_gate`
- `stage24_sampler_recomparison_gate`
- `stage24_single_active_sampler_gate`
- `stage24_reward_weights_unchanged_gate`
- `stage24_no_actor_leakage_gate`
- `stage25_base_protocol_gate`
- `stage25_train_eval_split_gate`
- `stage25_small_scale_mappo_pilot_gate`
- `stage25_reward_surface_alignment_gate`
- `stage25_visualization_report_gate`
- `stage25_torch_import_hygiene_gate`
- `stage25_no_forbidden_scaleup_or_architecture_gate`
- `stage26_harness_state_health_gate`
- `stage26_data_health_gate`
- `stage26_communication_health_gate`
- `stage26_consensus_health_gate`
- `stage26_reward_objective_health_gate`
- `stage26_assembler_health_gate`
- `stage26_sampler_health_gate`
- `stage26_actor_health_gate`
- `stage26_critic_health_gate`
- `stage26_mappo_loop_health_gate`
- `stage26_root_cause_matrix_gate`
- `stage26_no_training_or_tuning_gate`
- `stage27_critic_semantics_gate`
- `stage27_enriched_critic_features_gate`
- `stage27_return_normalization_gate`
- `stage27_critic_dataset_gate`
- `stage27_enriched_mlp_critic_gate`
- `stage27_graph_critic_gate`
- `stage27_critic_pretraining_gate`
- `stage27_critic_selection_gate`
- `stage27_no_actor_update_gate`
- `stage27_mappo_blocked_gate`
- `stage27_no_forbidden_architecture_gate`
- `actor_policy_input_schema_gate`
- `actor_policy_output_schema_gate`
- `training_precondition_gate`
- `post_task_self_review`

## Deferred Gates

- `credit_calibration_gate` - active before COMA, Q, direct edge-delta critic, or centralized critic work.

## Allowed Next Tasks

- `Stage 30` - reward/objective and projection alignment repair. Owner
  approval is required before executing Stage 30. This is a repair stage, not a
  larger policy-gradient pilot or scale-up training authorization.
- `Owner decision on Dec-POMDP feature boundary` - decide whether additional
  local history, local messages, or local resource summaries should be widened
  before any further evidence rebuild.

Stage 5 is closed. Do not open additional Stage 5.x planning tasks after the
Stage 5.10 exit gate unless a concrete defect is found in that exit gate.

Stage 6 is closed. Do not open additional Stage 6.x planning tasks after the
Stage 6 completion exit gate unless a concrete defect is found in that exit
gate.

Stage 7 is closed. Do not open additional Stage 7 planning tasks after the
Stage 7 completion exit gate unless a concrete defect is found in that gate.
Stage 7 completion includes multi-scenario topology evidence and a
manifest-validated evidence-only artifact.

Historical Stage 7 rule: Actor policy interface work is deferred to Stage 8
as recorded by the Stage 7 closeout.

Stage 29 is complete. Do not proceed into Stage 30, larger policy-gradient
pilots, scale-up training, checkpoint creation, sampler switching, reward
tuning, final tau selection, LSTM/recurrent PPO, or artifact writing outside
manifest-approved reports without owner approval. Stage 29 is decision-review
evidence only, not authorization for broader training.

## Blocked Tasks

- Unapproved training runs, hyperparameter sweeps, GPU workloads, and model
  checkpoints.
- Scale-up training, PPO/MAPPO reruns, and full Stage 11-15 reruns including
  the policy-gradient pilot.
- Generic evidence expansion that does not use Stage 18 disambiguated actor
  target semantics.
- Training execution without Stage 5.6 design contract and explicit owner approval.
- Training execution without Stage 5.7 architecture contract and explicit owner approval.
- Training execution without Stage 5.8 learning-target replay contract and explicit owner approval.
- Training execution without Stage 5.9 run-manifest artifact contract and explicit owner approval.
- Training execution without Stage 7 owner approval.
- Actor, critic, COMA, GNN, LSTM, or direct edge-delta implementation without
  Stage 7 owner approval.
- Actor policy interface without Stage 8 owner approval.
- Model implementation before Stage 8 policy interface contract.
- Model implementation before Stage 9 owner approval.
- Stage 26 work without owner approval.
- Stage 27 work without owner approval.
- Stage 28 work without owner approval.
- Stage 29 work without owner approval.
- Stage 30 work without owner approval.
- Larger policy-gradient pilots before Stage 30 repair and owner approval.
- Policy-gradient scale-up before Stage 30 repair and owner approval.
- Stage 9.1 work without owner approval.
- Stage 9 training execution or checkpoint creation.
- Local MLP training execution without owner approval.
- Local MLP checkpoint creation without owner approval.
- Training execution before Stage 8 policy interface and model approval.
- Dataset export without Stage 7 manifest guard and owner approval.
- Checkpoint creation before a future artifact writer contract.
- Additional Stage 5.x planning tasks after the Stage 5.10 exit gate.
- Additional Stage 6.x planning tasks after the Stage 6 completion exit gate.
- Additional Stage 7 planning tasks after the Stage 7 completion exit gate.
- Reward implementation beyond the Stage 5.2 pure interface skeleton, reward
  ablation, or active training objective integration.
- Reward weight calibration.
- Final `tau_consensus` selection without owner decision.
- Stage 4 PBFT consensus metric implementation before a Stage 4 contract.
- Stage 4 PBFT consensus implementation without owner approval.
- Consensus reward and application-level deadline objective implementation.
- v5 code migration, old reward copying, old metric-name adoption, or phase-script promotion.

## Recommended Next Task

`Stage 30 - reward/objective and projection alignment repair`.

Reason: Stage 29 determined that critic repair is no longer the main blocker,
but reward/objective alignment and projection friction remain blocking risks.
Stage 28 improved latency and energy while surrogate reward worsened, and top
proposal rejection worsened. The next actuator should repair or explicitly
diagnose that reward/projection coupling before any larger policy-gradient
pilot. Scale-up, checkpoint creation, reward-weight tuning, COMA, Transformer,
LSTM/recurrent PPO, sampler switching, and final tau selection remain blocked
until owner decision after Stage 30.

## Owner Decision Required

`true`

Codex may recommend this next task, but user approval is required before executing it.

## Active Goal Update - Stage 9 Forward-Only Model Scaffold

Controlled object: forward-only learnable model boundary from actor-safe local
edge tensors to actor edge scores, plus centralized training-only critic heads.

Desired state: `LocalMLPEdgeScorer` outputs logits/scores only; topology
selection stays in `ConflictAwareGreedyAssembler`; `CentralizedMLPCriticBaseline`
exposes value, feasibility, consensus-success-probability, latency, energy,
and add/remove/keep edge-delta heads without exporting critic outputs to actor
inputs.

State variables: actor feature fields, edge score records, critic global
feature fields, critic head names, no-training flags, and assembler projection
diagnostics.

Sensors: `tests/unit/test_stage9_forward_model_scaffold.py`,
`scripts/replay/stage9_0_model_forward_smoke_report.py`, full pytest, and
harness task validation.

Actuators: added `src/marl_topology/models/` tensorizers, MLP actor, MLP critic,
model registry, Stage 9 docs, smoke script, tests, and self-review report.

Disturbances: legacy activate schema confusion, centralized critic leakage into
deployment actor input, and accidental training/checkpoint routes.

Coupling map: actor-safe rows -> actor edge tensors -> edge logits ->
`EdgeScoreBatch` -> environment-side assembler; centralized evidence rows ->
critic tensors -> training-only critic heads.

Acceptance criteria: actor forward smoke pass, critic forward smoke pass,
assembler dry-run pass, active edge-score schema preserved, no optimizer,
backward, checkpoint, training artifact, COMA, PPO/MAPPO, GNN, GRU, LSTM,
Transformer, final tau selection, reward-weight calibration, or v5 migration.

Regression check: Stage 8 assembler remains the owner of hard topology
activation, and fixed top-k/full graph remain baselines rather than deployment
policy or oracle.

Residual risk: all Stage 9 models are untrained; Stage 10 must validate loss
alignment before any parameter update.

Recommended next task under active `/goal`: Stage 10 supervised loss dry-run,
only if Stage 9 verification passes.

Stage 9 verification result: passed.

- `python scripts\replay\stage9_0_model_forward_smoke_report.py`: actor
  forward, critic forward, and assembler dry-run passed with no artifact write.
- `python -m pytest -q`: `644 passed`.
- `python harness\scripts\validate_tasks.py`: task validation passed.

Next stage allowed under active `/goal`: Stage 10 supervised loss dry-run.

## Active Goal Update - Stage 10 Supervised Loss Dry-Run

Controlled object: supervised loss and target alignment between Stage 7
learning evidence, Stage 9 actor logits, and Stage 9 critic heads.

Desired state: finite actor and critic loss components are computed without
backward execution, optimizer creation, parameter update, checkpoint write, or
training artifact write.

State variables: actor edge labels, actor edge-delta diagnostic targets,
critic feasibility/consensus/latency/energy/value targets, critic edge-delta
targets, and parameter checksums before/after the dry-run.

Sensors: `tests/unit/test_stage10_supervised_loss_dry_run.py`,
`scripts/replay/stage10_supervised_loss_dry_run_report.py`, full pytest, and
harness task validation.

Actuators: added supervised batching and supervised loss modules, Stage 10
smoke script, documentation, tests, and post-task self-review.

Disturbances: target fields live near actor-safe rows in Stage 7 evidence, so
the batching layer must keep learning targets out of actor tensorization.

Coupling map: actor-safe rows -> actor tensors -> actor logits -> supervised
actor losses; centralized critic rows -> critic tensors -> critic heads ->
critic losses. Targets are loss-only and never actor inputs.

Acceptance criteria: finite scalar loss report, missing-target rejection,
actor-safe/target separation, and no backward, optimizer, checkpoint, artifact,
COMA, GNN, GRU, LSTM, PPO/MAPPO, Transformer, v5 migration, reward-weight
calibration, or final tau selection.

Regression check: Stage 9 forward-only model boundary remains no-update during
loss dry-run.

Residual risk: finite dry-run losses do not prove trainability; Stage 11 must
include tiny-batch overfit and validation diagnostics.

Recommended next task under active `/goal`: Stage 11 supervised Local MLP warm
start, only if Stage 10 verification passes.

Stage 10 verification result: passed.

- `python scripts\replay\stage10_supervised_loss_dry_run_report.py`: finite
  supervised loss components and no parameter update.
- `python -m pytest -q`: `648 passed`.
- `python harness\scripts\validate_tasks.py`: task validation passed.

Next stage allowed under active `/goal`: Stage 11 supervised Local MLP actor
warm start.

## Active Goal Update - Stage 11 Supervised Local MLP Actor Warm Start

Controlled object: Local MLP actor parameter state during supervised warm
start.

Desired state: only Local MLP actor parameters update under supervised edge
labels; critic training, RL updates, checkpoint writing, COMA, GNN, GRU, LSTM,
Transformer, final tau selection, reward-weight calibration, and v5 migration
remain inactive.

State variables: train/validation BCE and ranking diagnostics, precision,
recall, pairwise accuracy, Spearman diagnostic, tiny-batch overfit loss,
parameter checksums, update-step count, assembler feasibility/latency/energy
diagnostics, and manifest validation result.

Sensors: `tests/unit/test_stage11_supervised_actor_trainer.py`,
`scripts/train/stage11_supervised_mlp_actor.py`, full pytest, and harness task
validation.

Actuators: added actor-only supervised trainer, Stage 11 train script,
documentation, tests, manifest-validated optional report artifact path, and
self-review.

Disturbances: Stage 7 topology variants can give contradictory labels for the
same actor-local edge observations; tiny-batch overfit is therefore measured on
a deterministic mixed-label row, and full-dataset metrics remain diagnostic.

Coupling map: Stage 7 actor-safe rows -> local edge tensors -> supervised MLP
actor update -> actor scores -> Stage 8 recommended assembler ->
environment-side topology evaluation.

Acceptance criteria: actor-only training completes, tiny-batch overfit passes,
assembler consumes trained actor scores, manifest validation passes, default
smoke run writes no artifact, no checkpoint is written, and no critic/RL
training path is active.

Regression check: deployment actor input remains actor-safe only, and
environment-side assembler remains hard topology owner.

Residual risk: supervised MLP quality is limited by small contradictory
evidence; Stage 12 must test centralized critic and edge-delta fidelity before
any policy-gradient pilot.

Recommended next task under active `/goal`: Stage 12 centralized critic and
edge-delta pretraining, only if Stage 11 verification passes.

Stage 11 verification result: passed.

- `python scripts\train\stage11_supervised_mlp_actor.py`: actor-only
  supervised update completed, tiny-batch overfit passed, manifest validation
  passed, default smoke wrote no artifact and no checkpoint.
- `python -m pytest -q`: `652 passed`.
- `python harness\scripts\validate_tasks.py`: task validation passed.

Next stage allowed under active `/goal`: Stage 12 centralized critic and
edge-delta pretraining.

## Active Goal Update - Stage 12 Centralized Critic And Edge-Delta Pretraining

Controlled object: centralized MLP critic parameter state and direct
add/remove/keep edge-delta heads.

Desired state: only critic parameters update; actor and RL paths remain
inactive; the stage produces fidelity evidence for scalar heads and edge-delta
heads before any policy-gradient pilot.

State variables: critic loss before/after, critic parameter checksums,
add/remove edge-delta ranking metrics, balanced sign accuracy, helpful-edge
precision, harmful-edge recall, feasibility accuracy, consensus/latency/energy
errors, calibration errors, and collapse diagnostics.

Sensors: `tests/unit/test_stage12_critic_pretrainer.py`,
`scripts/train/stage12_critic_pretraining.py`, full pytest, and harness task
validation.

Actuators: added critic pretrainer, Stage 12 script, documentation, tests, and
self-review.

Disturbances: small imbalanced Stage 7 evidence can make edge-delta fidelity
look better or worse than real deployment credit quality.

Coupling map: centralized Stage 7 training views -> critic tensors -> MLP
critic heads -> scalar and edge-delta losses -> fidelity report. Critic outputs
remain training-only and cannot enter actor inputs.

Acceptance criteria: critic-only update completes, fidelity report produced,
non-collapsed predictions, basic sign/rank sanity for direct edge-delta heads,
no actor RL fine-tune, no COMA, no GNN/temporal/Transformer actor, no
checkpoint, no final tau selection, no reward-weight calibration, and no v5
migration.

Regression check: deployment actor input remains local-only and independent of
critic outputs.

Residual risk: Stage 12 fidelity is a small-sample sanity gate, not final proof
of credit assignment.

Recommended next task under active `/goal`: Stage 13 local GNN edge scorer
comparison, only if Stage 12 verification passes.

Stage 12 verification result: passed.

- `python scripts\train\stage12_critic_pretraining.py`: critic-only update
  completed, fidelity gate passed, no actor/RL update, no checkpoint, no
  artifact write.
- `python -m pytest -q`: `655 passed`.
- `python harness\scripts\validate_tasks.py`: task validation passed.

Next stage allowed under active `/goal`: Stage 13 local GNN edge scorer
comparison.

## Active Goal Update - Stage 13 Local GNN Edge Scorer

Controlled object: actor architecture comparison between supervised Local MLP
and local ego-graph edge scorer.

Desired state: `LocalGNNEdgeScorer` uses only per-actor local incident
candidate-edge groups, outputs edge scores only, and is compared with the MLP
baseline on validation loss, ranking diagnostics, assembler diagnostics,
runtime, and parameter count.

State variables: grouped actor tensor ids, validation edge loss, pairwise
accuracy, assembler feasibility/latency/energy diagnostics, runtime,
parameter count, local graph scope, and leakage flags.

Sensors: `tests/unit/test_stage13_local_gnn_edge_scorer.py`,
`scripts/train/stage13_supervised_gnn_actor.py`, full pytest, and harness task
validation.

Actuators: added grouped actor tensor support, local GNN model, comparison
script, documentation, tests, and self-review.

Disturbances: small contradictory Stage 7 labels may cause GNN to underperform
MLP; this is reported as an architecture diagnostic rather than hidden.

Coupling map: actor-safe rows -> grouped local edge tensors -> local GNN or MLP
actor scores -> Stage 8 assembler -> environment-side diagnostics.

Acceptance criteria: local GNN implemented and tested, comparison report
produced, no global actor graph or oracle/critic leakage, no GRU/LSTM,
Transformer, PPO/MAPPO, COMA, checkpoint, final tau selection, reward-weight
calibration, or v5 migration, and a provisional Stage 14 starting actor is
selected with a reason.

Regression check: Dec-POMDP actor boundary remains local-only.

Residual risk: GNN evidence is limited by current supervised data diversity.

Recommended next task under active `/goal`: Stage 14 GRU/LSTM temporal actor
ablation, only if Stage 13 verification passes.

Stage 13 verification result: passed.

- `python scripts\train\stage13_supervised_gnn_actor.py`: local GNN comparison
  completed; GNN improved validation loss on the current evidence and was
  selected as the provisional Stage 14 starting actor.
- `python -m pytest -q`: `659 passed`.
- `python harness\scripts\validate_tasks.py`: task validation passed.

Next stage allowed under active `/goal`: Stage 14 temporal actor ablation.

## Active Goal Update - Stage 14 Temporal GRU/LSTM Actor Ablation

Controlled object: actor-safe temporal memory for local edge scoring.

Desired state: sequence batching preserves local time order; GRU is tested
first; LSTM is tested only after GRU sanity passes; temporal models consume
only local edge histories and output edge scores.

State variables: temporal feature tensors, target tensors, masks, sequence ids,
time steps, GRU/LSTM losses, output-collapse flags, hidden reset states,
future-leakage flag, and selected Stage 15 actor.

Sensors: `tests/unit/test_stage14_temporal_actor_ablation.py`,
`scripts/train/stage14_temporal_actor_ablation.py`, full pytest, and harness
task validation.

Actuators: added sequence batching, LocalGRUEdgeScorer, LocalLSTMEdgeScorer,
temporal ablation script, documentation, tests, and self-review.

Disturbances: real Stage 7 evidence has only time step 0, so temporal testing
uses an in-memory actor-safe fixture and does not promote recurrent models for
Stage 15 deployment.

Coupling map: actor-safe local edge features -> ordered edge sequences ->
GRU/LSTM edge-score outputs. Environment-side assembler ownership remains
unchanged.

Acceptance criteria: sequence time-order sensor passes, future leakage sensor
passes, hidden reset works, variable mask works, GRU tested before LSTM, LSTM
tested after GRU sanity, no global memory, no Transformer, no PPO/MAPPO, no
COMA, no checkpoint, no final tau selection, no reward-weight calibration, and
no v5 migration.

Regression check: deployment actor memory remains local-history-only.

Residual risk: temporal evidence is synthetic fixture-level only; real
multi-step evidence is still needed before recurrent deployment selection.

Recommended next task under active `/goal`: Stage 15 controlled policy-gradient
pilot, only if Stage 14 verification passes.

Stage 14 verification result: passed.

- `python scripts\train\stage14_temporal_actor_ablation.py`: GRU sanity
  passed, LSTM tested after GRU, no future leakage detected, no checkpoint or
  artifact write.
- `python -m pytest -q`: `664 passed`.
- `python harness\scripts\validate_tasks.py`: task validation passed.

Next stage allowed under active `/goal`: Stage 15 controlled policy-gradient
pilot.

## Active Goal Update - Stage 15 Controlled Policy-Gradient Pilot

Controlled object: short online fine-tune pilot for the Stage 13 local GNN
actor using clipped proposal-policy updates and assembler projection.

Desired state: run a small fixed-seed pilot or stop with a reason; record
before/after supervised actor diagnostics, proposal/projection log-probability
semantics, objective diagnostics, projection rejection rate, KL/entropy/clip
metrics, value-loss diagnostic, actor score distribution, and safety stop
flags.

State variables: consensus_success_probability, violation rate under
`tau_requirement_min = 0.9`, latency, energy, topology diagnostics, edge count,
policy loss, value loss, KL proxy, entropy, clip fraction, actor score
statistics, projection rejection rate, collapse flag, and safety degradation
flag.

Sensors: `tests/unit/test_stage15_controlled_policy_pilot.py`,
`scripts/train/stage15_controlled_ppo_pilot.py`, full pytest, and harness task
validation.

Actuators: added controlled policy pilot module, Stage 15 script,
documentation, tests, and self-review.

Disturbances: the pilot is a single-fixture smoke run and uses an explicit
non-calibrated pilot surrogate config; it cannot support convergence claims.

Coupling map: local GNN actor logits -> sampled directed-edge proposal
indicators -> assembler-projected topology -> evaluator metrics -> Stage 5.2
training-only surrogate signal -> clipped proposal-policy update. Projected
topology log-probability is not claimed exact.

Acceptance criteria: pilot completes or stops with reason, before/after
comparison reported, projection diagnostics logged, no COMA, no Transformer,
no large sweep, no final tau selection, no reward-weight calibration, no
checkpoint, no v5 migration, and post-task self-review produced.

Regression check: actor inputs remain local-only and assembler projection
remains environment-side.

Residual risk: Stage 15 is a pilot only; Stage 16 requires explicit owner
decision and stronger evidence.

Recommended next task under active `/goal`: write final Stage 9-15 stack review
and stop before Stage 16.

Stage 15 verification result: passed.

- `python scripts\train\stage15_controlled_ppo_pilot.py`: pilot completed,
  violation rate did not worsen, actor did not collapse, projection rejection
  rate remained 0.0, no checkpoint or artifact write.
- `python -m pytest -q`: `667 passed`.
- `python harness\scripts\validate_tasks.py`: task validation passed.

Next stage allowed under active `/goal`: final Stage 9-15 stack review only.
Stage 16 is not authorized and requires owner decision.
