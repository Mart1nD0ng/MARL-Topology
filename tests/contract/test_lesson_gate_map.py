from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_lesson_to_gate_map_references_required_gates_and_lessons() -> None:
    text = (ROOT / "docs" / "V5_LESSON_TO_GATE_MAP.md").read_text(encoding="utf-8")

    required_terms = [
        "metric_governance_gate",
        "consensus_protocol_naming_gate",
        "full_mask_not_oracle_gate",
        "oracle_before_infeasible_gate",
        "fixed_threshold_is_baseline_gate",
        "reward_plateau_resource_gate",
        "dec_pomdp_leakage_gate",
        "credit_calibration_gate",
        "phase_script_entropy_gate",
        "physics_regime_declaration_gate",
        "L001_metric_governance_first",
        "L002_effective_success_alias_risk",
        "L003_full_mask_not_oracle",
        "L004_reliability_plateau_before_resource_objective",
        "L005_oracle_before_infeasible_claims",
        "L006_fixed_threshold_deployment_risk",
        "L007_credit_calibration_not_impossibility",
        "L008_phase_script_entropy",
        "L009_dec_pomdp_boundary_before_actor",
        "L012_physics_can_mask_learning_claims",
    ]
    missing = [term for term in required_terms if term not in text]
    assert not missing, f"lesson-to-gate map missing required terms: {missing}"


def test_stage2_harness_tasks_exist_and_name_lesson_gates() -> None:
    task_names = [
        "implement_goal_skeleton_v0.yaml",
        "review_scene3d_candidate_graph.yaml",
        "review_link_model_contract.yaml",
        "review_topology_evaluator_contract.yaml",
    ]

    for name in task_names:
        path = ROOT / "harness" / "tasks" / name
        assert path.exists(), f"missing Stage 2 harness task: {name}"
        task = yaml.safe_load(path.read_text(encoding="utf-8"))
        for field in [
            "relevant_lessons",
            "required_tests",
            "forbidden_v5_inheritance",
            "expected_outputs",
            "negative_checks",
        ]:
            assert field in task, f"{name} missing {field}"
            assert task[field], f"{name} has empty {field}"


def test_metric_governance_defaults_are_not_reintroduced() -> None:
    files = [
        ROOT / "docs" / "METRIC_CONTRACT.md",
        ROOT / "docs" / "V5_LESSON_TO_GATE_MAP.md",
    ]
    offenders: dict[str, list[str]] = {}
    banned = ["`P_eff_soft`", "`P_eff_hard`"]

    for path in files:
        text = path.read_text(encoding="utf-8")
        hits = [term for term in banned if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"old P_eff soft/hard defaults reintroduced: {offenders}"
