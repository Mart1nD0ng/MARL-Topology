from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


STAGE3_CONTRACTS = [
    "GEOMETRY3D_CONTRACT.md",
    "CHANNEL_MODEL_CONTRACT.md",
    "LINK_TRANSMISSION_CONTRACT.md",
    "NETWORK_LAYER_CONTRACT.md",
]


def test_stage3_plan_exists_and_freezes_layer_order() -> None:
    text = (ROOT / "docs" / "STAGE3_COMMUNICATION_SIMULATION_PLAN.md").read_text(
        encoding="utf-8"
    )

    required = [
        "Stage 3.0 Communication Simulation Plan",
        "3D Environment / Scenario",
        "Geometry / Visibility",
        "Channel Simulation",
        "Link Layer Transmission",
        "Network Layer Communication",
        "Stage 3 must not define",
        "Selected Edge Semantics",
        "Full graph remains a baseline topology, not an oracle",
        "Stage 3.1 - 3D city scenario and geometry visibility",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 3 plan missing terms: {missing}"


def test_stage3_plan_lists_required_micro_fixtures() -> None:
    text = (ROOT / "docs" / "STAGE3_COMMUNICATION_SIMULATION_PLAN.md").read_text(
        encoding="utf-8"
    )

    fixtures = [
        "free_space_close",
        "free_space_far",
        "blocked_by_building",
        "urban_gap_los",
        "urban_canyon_nlos",
        "rsu_high_los",
        "two_transmitters_interference",
        "orthogonal_channel_no_interference",
        "multi_hop_delivery",
        "resource_redundant_topology",
    ]
    missing = [fixture for fixture in fixtures if fixture not in text]
    assert not missing, f"Stage 3 plan missing micro-fixtures: {missing}"


def test_stage3_contracts_have_required_sections_and_stage4_deferral() -> None:
    required_sections = [
        "## Responsibility",
        "## Inputs",
        "## Outputs",
        "## Units",
        "## Assumptions",
        "## Omitted Components",
        "## Tests",
        "## Failure Modes",
        "## GOAL_SKELETON Coupling",
        "## Deferred To Stage 4",
    ]

    missing_by_doc: dict[str, list[str]] = {}
    for name in STAGE3_CONTRACTS:
        text = (ROOT / "docs" / name).read_text(encoding="utf-8")
        missing = [section for section in required_sections if section not in text]
        if missing:
            missing_by_doc[name] = missing

    assert not missing_by_doc, f"Stage 3 contracts missing sections: {missing_by_doc}"


def test_stage3_contracts_do_not_define_application_consensus_metrics() -> None:
    forbidden_snake_names = [
        "consensus_success",
        "consensus_success_probability",
        "P_eff_soft",
        "P_eff_hard",
        "hard_eval",
        "soft_train",
    ]
    offenders: dict[str, list[str]] = {}
    for name in ["STAGE3_COMMUNICATION_SIMULATION_PLAN.md", *STAGE3_CONTRACTS]:
        text = (ROOT / "docs" / name).read_text(encoding="utf-8")
        hits = [term for term in forbidden_snake_names if term in text]
        if hits:
            offenders[name] = hits

    assert not offenders, f"Stage 3 contracts define forbidden consensus metrics: {offenders}"


def test_stage3_harness_tasks_exist() -> None:
    task_ids = [
        "stage3_communication_simulation_plan",
        "stage3_geometry_visibility_review",
        "stage3_channel_model_review",
        "stage3_link_transmission_review",
        "stage3_network_layer_review",
        "stage3_micro_fixture_suite_review",
    ]

    for task_id in task_ids:
        path = ROOT / "harness" / "tasks" / f"{task_id}.yaml"
        task = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert task["id"] == task_id
        for field in [
            "required_artifacts",
            "expected_evidence",
            "negative_checks",
            "relevant_lessons",
            "required_tests",
            "forbidden_v5_inheritance",
            "expected_outputs",
        ]:
            assert field in task
            assert task[field]


def test_project_state_marks_stage3_0_complete_and_blocks_stage4() -> None:
    text = (ROOT / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")

    required = [
        "stage_3_0_communication_simulation_design",
        "stage3_communication_simulation_gate",
        "stage3_geometry_visibility_gate",
        "stage3_channel_model_gate",
        "stage3_link_transmission_gate",
        "stage3_network_layer_gate",
        "stage_4_pbft_consensus_metric_implementation_before_contract",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 3.0 state: {missing}"
