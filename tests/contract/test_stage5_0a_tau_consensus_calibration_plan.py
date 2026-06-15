from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_tau_consensus_calibration_plan_exists_and_does_not_select_tau() -> None:
    text = _read_doc("TAU_CONSENSUS_CALIBRATION_PLAN.md")

    required = [
        "Tau Consensus Calibration Plan",
        "Stage 5.0a Scope",
        "does not choose the final threshold",
        "tau_consensus",
        "consensus_success_probability >= tau_consensus",
        "Stage 4.8's diagnostic `reliability_threshold = 0.2` remains non-authoritative",
        "Stage 4.8 can be used as a smoke-test sensor only",
        "Stage 5.0a intentionally does not provide default numeric tau candidates",
        "owner-approved reliability threshold",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"TAU_CONSENSUS_CALIBRATION_PLAN missing terms: {missing}"


def test_tau_plan_requires_scenario_coverage_and_registered_inputs() -> None:
    text = _read_doc("TAU_CONSENSUS_CALIBRATION_PLAN.md")

    required = [
        "registered `consensus_success_probability`",
        "registered `latency`",
        "registered `energy`",
        "`topology_diagnostics`",
        "urlcc_finite_blocklength_v1",
        "pbft_expected_initiator_mean_field_v1",
        "clear free-space communication",
        "near-threshold communication",
        "blocked or NLoS urban communication",
        "same-resource interference",
        "orthogonal-resource mitigation",
        "sparse topology candidates",
        "dense/full graph baseline candidates",
        "failed scheduled-message cases",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"tau plan missing calibration inputs or coverage: {missing}"


def test_tau_plan_forbids_threshold_copying_and_reward_shortcuts() -> None:
    text = _read_doc("TAU_CONSENSUS_CALIBRATION_PLAN.md")

    required = [
        "copy Stage 4.8 `reliability_threshold = 0.2` as `tau_consensus`",
        "use `P_eff`, hard/soft legacy naming",
        "label full graph as oracle",
        "use policy failure as infeasibility proof",
        "introduce reward implementation",
        "run training",
        "implement actor, critic, COMA, GNN, or LSTM code",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"tau plan missing forbidden shortcuts: {missing}"


def test_stage5_contracts_reference_tau_calibration_plan() -> None:
    objective = _read_doc("OBJECTIVE_CONTRACT.md")
    reward = _read_doc("REWARD_CONTRACT.md")
    surrogate = _read_doc("REWARD_SURROGATE_CONTRACT.md")
    metric = _read_doc("METRIC_CONTRACT.md")
    freeze = _read_doc("STAGE5_REWARD_OBJECTIVE_FREEZE.md")

    for name, text in {
        "OBJECTIVE_CONTRACT": objective,
        "REWARD_CONTRACT": reward,
        "REWARD_SURROGATE_CONTRACT": surrogate,
        "METRIC_CONTRACT": metric,
        "STAGE5_REWARD_OBJECTIVE_FREEZE": freeze,
    }.items():
        assert "Stage 5.0a" in text, f"{name} missing Stage 5.0a reference"
    assert "TAU_CONSENSUS_CALIBRATION_PLAN.md" in objective
    assert "TAU_CONSENSUS_CALIBRATION_PLAN.md" in reward
    assert "does not select the final `tau_consensus`" in objective
    assert "does not set a final `tau_consensus`" in reward
    assert "Stage 5.0a does not add metric names" in metric


def test_stage5_0a_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage5_0a_tau_consensus_calibration_plan.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage5_0a_tau_consensus_calibration_plan"
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
    assert "reliability_threshold 0.2" in negative_text
    assert "final tau_consensus" in negative_text
    assert "v5" in negative_text


def test_project_state_marks_stage5_0a_complete_and_keeps_work_blocked() -> None:
    text = _read_doc("PROJECT_STATE.md")

    required = [
        "post_stage_5_0a_awaiting_owner_decision",
        "stage_5_0a_tau_consensus_calibration_plan",
        "stage5_0a_tau_consensus_calibration_plan_gate",
        "reward_implementation",
        "reward_weight_calibration",
        "training_runs",
        "recommended_next_task: stage_5_0c_tau_consensus_calibration_report_design",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 5.0a state: {missing}"


def test_stage5_0a_does_not_add_reward_training_or_model_code() -> None:
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

    reward_paths = [
        path
        for path in (ROOT / "src" / "marl_topology").rglob("*.py")
        if "reward" in path.name.lower()
        and path.relative_to(ROOT).as_posix()
        != "src/marl_topology/evaluation/reward_surface_analysis.py"
    ]

    assert not offenders, f"Stage 5.0a added forbidden code: {offenders}"
    assert not reward_paths, f"Stage 5.0a added reward implementation files: {reward_paths}"


def test_tau_plan_does_not_register_new_metrics_or_legacy_aliases() -> None:
    text = _read_doc("TAU_CONSENSUS_CALIBRATION_PLAN.md")
    metric = _read_doc("METRIC_CONTRACT.md")

    assert "No unregistered metric may enter the tau selection table" in text
    assert "Stage 5.0a does not add metric names" in metric
    banned_defaults = [
        "`P_eff_soft`",
        "`P_eff_hard`",
        "`hard_eval`",
        "`soft_train`",
    ]
    offenders = [term for term in banned_defaults if term in text or term in metric]
    assert not offenders, f"Stage 5.0a reintroduced legacy defaults: {offenders}"
