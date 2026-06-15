from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_tau_calibration_report_design_exists_and_does_not_select_tau() -> None:
    text = _read_doc("TAU_CONSENSUS_CALIBRATION_REPORT_DESIGN.md")

    required = [
        "Tau Consensus Calibration Report Design",
        "Stage 5.0c Scope",
        "does not run calibration",
        "does not choose `tau_consensus`",
        "Boundary shorthand: Stage 5.0c does not select the final threshold",
        "decision-support sensor",
        "not a reward implementation",
        "not a training artifact",
        "not an actor input source",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"report design missing boundary terms: {missing}"


def test_report_design_lists_required_input_collections() -> None:
    text = _read_doc("TAU_CONSENSUS_CALIBRATION_REPORT_DESIGN.md")

    required = [
        "Scenario Set Manifest",
        "Topology Evaluation Rows",
        "Candidate Tau Rows",
        "Protocol And Regime Rows",
        "`scenario_set_id`",
        "`scenario_family`",
        "`physics_regime`",
        "`link_transmission_regime`",
        "`pbft_model_id`",
        "`consensus_success_probability`",
        "`latency`",
        "`energy`",
        "`tau_candidate`",
        "`tau_source`",
        "`is_default`",
        "`is_selected`",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"report design missing input schema terms: {missing}"


def test_report_design_lists_output_tables_and_decision_packet() -> None:
    text = _read_doc("TAU_CONSENSUS_CALIBRATION_REPORT_DESIGN.md")

    required = [
        "Scenario Summary",
        "Tau Feasibility Summary",
        "Topology By Tau Detail",
        "Owner Decision Packet",
        "`feasible_topology_count`",
        "`feasible_non_full_topology_count`",
        "`full_graph_only_feasible`",
        "`lowest_feasible_latency`",
        "`lowest_feasible_energy`",
        "`pareto_feasible_count`",
        "`recommended_tau_candidate`",
        "`owner_selected_tau`",
        "`owner_decision_status`",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"report design missing output table terms: {missing}"


def test_report_design_forbids_default_tau_and_stage48_threshold_copy() -> None:
    text = _read_doc("TAU_CONSENSUS_CALIBRATION_REPORT_DESIGN.md")

    required = [
        "does not provide default numeric tau candidates",
        "Stage 4.8 `reliability_threshold = 0.2` appears as an implicit default",
        "No final tau is selected",
        "No default numeric tau candidate is introduced",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"report design missing tau negative checks: {missing}"


def test_report_design_metric_governance_and_validation_gates() -> None:
    text = _read_doc("TAU_CONSENSUS_CALIBRATION_REPORT_DESIGN.md")

    required = [
        "Stage 5.0c adds no metric names",
        "`consensus_success_probability`",
        "`latency`",
        "`energy`",
        "`topology_diagnostics`",
        "full graph is labeled oracle",
        "oracle-candidate labels are marked as deployment actor input",
        "`network_delivery_probability` is renamed as",
        "reward, training, actor, critic, COMA, GNN, LSTM, or v5 code is introduced",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"report design missing metric or validation gates: {missing}"


def test_stage5_contracts_reference_report_design() -> None:
    docs = {
        "TAU_CONSENSUS_CALIBRATION_PLAN": _read_doc("TAU_CONSENSUS_CALIBRATION_PLAN.md"),
        "OBJECTIVE_CONTRACT": _read_doc("OBJECTIVE_CONTRACT.md"),
        "REWARD_CONTRACT": _read_doc("REWARD_CONTRACT.md"),
        "REWARD_SURROGATE_CONTRACT": _read_doc("REWARD_SURROGATE_CONTRACT.md"),
        "METRIC_CONTRACT": _read_doc("METRIC_CONTRACT.md"),
        "STAGE5_REWARD_OBJECTIVE_FREEZE": _read_doc("STAGE5_REWARD_OBJECTIVE_FREEZE.md"),
    }

    for name, text in docs.items():
        assert "Stage 5.0c" in text, f"{name} missing Stage 5.0c reference"
    assert "TAU_CONSENSUS_CALIBRATION_REPORT_DESIGN.md" in docs["TAU_CONSENSUS_CALIBRATION_PLAN"]
    assert "TAU_CONSENSUS_CALIBRATION_REPORT_DESIGN.md" in docs["OBJECTIVE_CONTRACT"]
    assert "TAU_CONSENSUS_CALIBRATION_REPORT_DESIGN.md" in docs["REWARD_CONTRACT"]
    assert "Stage 5.0c does not add metric names" in docs["METRIC_CONTRACT"]


def test_stage5_0c_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage5_0c_tau_consensus_calibration_report_design.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage5_0c_tau_consensus_calibration_report_design"
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
    negative_text = " ".join(task["negative_checks"])
    assert "default numeric tau candidates" in negative_text
    assert "reliability_threshold 0.2" in negative_text
    assert "v5" in negative_text


def test_project_state_marks_stage5_0c_complete_and_recommends_report_only_implementation() -> None:
    text = _read_doc("PROJECT_STATE.md")

    required = [
        "post_stage_5_0c_awaiting_owner_decision",
        "stage_5_0c_tau_consensus_calibration_report_design",
        "stage5_0c_tau_consensus_calibration_report_design_gate",
        "stage_5_0d_tau_consensus_calibration_report_implementation_without_tau_selection",
        "recommended_next_task: stage_5_0d_tau_consensus_calibration_report_implementation_without_tau_selection",
        "reward_implementation",
        "reward_weight_calibration",
        "training_runs",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 5.0c state: {missing}"


def test_stage5_0c_boundary_does_not_add_reward_training_or_model_code() -> None:
    banned_terms = [
        "class Actor(",
        "class Critic(",
        "COMA",
        "MAPPO",
        "optimizer",
        "train_loop",
        "def reward",
        "def compute_reward",
        "class Reward",
        "reward =",
        "reward:",
        "P_eff_soft",
        "P_eff_hard",
        "hard_eval",
        "soft_train",
        "D:\\PhD_works\\v5",
    ]
    offenders: dict[str, list[str]] = {}
    for path in (ROOT / "src" / "marl_topology").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        hits = [term for term in banned_terms if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"Stage 5.0c added forbidden code: {offenders}"
    design_text = _read_doc("TAU_CONSENSUS_CALIBRATION_REPORT_DESIGN.md")
    assert "The executable report is not implemented in Stage 5.0c" in design_text
