from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_objective_contract_exists_and_freezes_objective_semantics() -> None:
    text = _read_doc("OBJECTIVE_CONTRACT.md")

    required = [
        "Objective Contract",
        "Stage 5.0 Objective Freeze",
        "consensus_success_probability",
        "protocol_latency",
        "protocol_energy",
        "topology_diagnostics",
        "consensus_success_probability >= tau_consensus",
        "minimize latency",
        "minimize energy",
        "Stage 4.8 used `reliability_threshold = 0.2` only as an audit diagnostic",
        "not the formal `tau_consensus`",
        "Stage 5.0 does not choose latency/energy weights",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"OBJECTIVE_CONTRACT missing terms: {missing}"


def test_reward_surrogate_contract_exists_and_blocks_implementation() -> None:
    text = _read_doc("REWARD_SURROGATE_CONTRACT.md")

    required = [
        "Reward Surrogate Contract",
        "future training surrogate only",
        "Stage 5.0 does not implement reward code",
        "smooth constraint-violation penalty",
        "latency penalty",
        "energy penalty",
        "reliability plateau",
        "scale normalization",
        "Stage 4.8 is a boundary audit, not a calibration set",
        "Reward implementation remains blocked",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"REWARD_SURROGATE_CONTRACT missing terms: {missing}"


def test_stage5_freeze_doc_records_deferred_work_and_no_training_boundary() -> None:
    text = _read_doc("STAGE5_REWARD_OBJECTIVE_FREEZE.md")

    required = [
        "Stage 5.0 Reward / Objective Contract Freeze",
        "does not implement reward",
        "does not train models",
        "tau_consensus",
        "not `tau_consensus`",
        "Deferred Work",
        "reward implementation",
        "reward weight calibration",
        "training runs",
        "actor, critic, COMA, GNN, or LSTM implementation",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"STAGE5_REWARD_OBJECTIVE_FREEZE missing terms: {missing}"


def test_reward_and_metric_contracts_record_stage5_freeze() -> None:
    reward = _read_doc("REWARD_CONTRACT.md")
    metric = _read_doc("METRIC_CONTRACT.md")

    reward_required = [
        "Stage 5.0 Reward / Objective Contract Freeze",
        "does not implement a reward function",
        "docs/OBJECTIVE_CONTRACT.md",
        "docs/REWARD_SURROGATE_CONTRACT.md",
        "tau_consensus",
        "reliability_threshold = 0.2",
        "not the formal `tau_consensus`",
        "standalone timeout reward",
        "standalone quorum reward",
        "standalone density reward",
        "reliability bonus above `tau_consensus`",
    ]
    metric_required = [
        "Stage 5.0 Objective / Reward Contract Freeze",
        "does not add metric names",
        "`protocol_latency` and `protocol_energy` are source quantities",
        "`tau_consensus` is an objective-contract parameter, not a metric name",
        "not the formal `tau_consensus`",
    ]
    missing_reward = [item for item in reward_required if item not in reward]
    missing_metric = [item for item in metric_required if item not in metric]

    assert not missing_reward, f"REWARD_CONTRACT missing Stage 5.0 terms: {missing_reward}"
    assert not missing_metric, f"METRIC_CONTRACT missing Stage 5.0 terms: {missing_metric}"


def test_stage5_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage5_reward_objective_contract_freeze.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage5_reward_objective_contract_freeze"
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
    assert "reward implementation" in negative_text
    assert "unregistered metric" in negative_text


def test_project_state_marks_stage5_complete_and_keeps_implementation_blocked() -> None:
    text = _read_doc("PROJECT_STATE.md")

    required = [
        "post_stage_5_0_awaiting_owner_decision",
        "stage_5_0_reward_objective_contract_freeze",
        "stage5_reward_objective_contract_freeze_gate",
        "reward_implementation",
        "reward_weight_calibration",
        "recommended_next_task: stage_5_0a_tau_consensus_calibration_plan",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 5.0 state: {missing}"


def test_stage5_does_not_add_reward_training_or_model_code() -> None:
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

    assert not offenders, f"Stage 5.0 added forbidden code: {offenders}"
    assert not reward_paths, f"Stage 5.0 added reward implementation files: {reward_paths}"


def test_forbidden_reward_terms_are_documented_as_forbidden_not_active() -> None:
    objective = _read_doc("OBJECTIVE_CONTRACT.md")
    surrogate = _read_doc("REWARD_SURROGATE_CONTRACT.md")

    for text in (objective, surrogate):
        assert "Forbidden" in text
        for term in [
            "standalone timeout reward",
            "standalone quorum reward",
            "standalone density reward",
            "standalone edge-count reward",
            "reliability bonus above `tau_consensus`",
            "unregistered metrics",
        ]:
            assert term in text


def test_metric_governance_still_uses_existing_registered_metrics_only() -> None:
    objective = _read_doc("OBJECTIVE_CONTRACT.md")
    metric = _read_doc("METRIC_CONTRACT.md")

    for name in [
        "consensus_success",
        "consensus_success_probability",
        "latency",
        "energy",
        "topology_diagnostics",
    ]:
        assert name in objective
        assert name in metric
    assert "Stage 5.0 adds no new metric names" in metric
