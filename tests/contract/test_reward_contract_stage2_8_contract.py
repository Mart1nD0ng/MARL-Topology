from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_stage_2_8_reward_review_doc_exists_and_blocks_implementation() -> None:
    text = (ROOT / "docs" / "REWARD_CONTRACT_REVIEW.md").read_text(encoding="utf-8")

    required = [
        "Stage 2.8 Reward Contract Review",
        "There is no reward code in this stage.",
        "No active reward function exists.",
        "No training surrogate is registered.",
        "No new metric is registered.",
        "No project-wide reliability threshold `tau` is fixed.",
        "consensus_success",
        "consensus_success_probability",
        "latency",
        "energy",
        "topology_diagnostics",
        "timeout or quorum as standalone reward terms",
        "full graph is baseline",
        "actor and replay projections exclude reward",
        "v5 remains a read-only experience library",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"REWARD_CONTRACT_REVIEW missing terms: {missing}"


def test_reward_contract_records_stage_2_8_no_active_reward_boundary() -> None:
    text = (ROOT / "docs" / "REWARD_CONTRACT.md").read_text(encoding="utf-8")

    required = [
        "Stage 2.8 Reward Contract Review",
        "no active reward",
        "no reward implementation module",
        "no reward weights",
        "no training surrogate",
        "no reward column admitted into replay datasets",
        "Reliability remains a constraint",
        "Latency and energy remain the objectives",
        "No project-wide reliability threshold `tau` is fixed",
        "Required Reward-Hacking Tests",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"REWARD_CONTRACT missing Stage 2.8 terms: {missing}"


def test_stage_2_8_does_not_reintroduce_legacy_metric_defaults() -> None:
    paths = [
        ROOT / "docs" / "REWARD_CONTRACT.md",
        ROOT / "docs" / "REWARD_CONTRACT_REVIEW.md",
    ]
    banned_defaults = [
        "`P_eff_soft`",
        "`P_eff_hard`",
        "`hard_eval`",
        "`soft_train`",
    ]
    offenders: dict[str, list[str]] = {}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        hits = [term for term in banned_defaults if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"Stage 2.8 reintroduced legacy metric defaults: {offenders}"


def test_stage_2_8_reward_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "reward_contract_review_stage2_8.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    for field in [
        "relevant_lessons",
        "required_tests",
        "forbidden_v5_inheritance",
        "expected_outputs",
        "negative_checks",
    ]:
        assert field in task
        assert task[field]

    negative_text = " ".join(task["negative_checks"])
    for expected in ["old v5 reward", "P_eff", "timeout quorum", "ActorObservation", "training"]:
        assert expected in negative_text


def test_project_state_marks_stage_2_8_complete_and_waits_for_owner() -> None:
    text = (ROOT / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")

    required = [
        "stage_2_8_reward_contract_review_without_implementation",
        "reward_contract_review_gate",
        "reward_plateau_resource_gate",
        "training_precondition_gate",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 2.8 state: {missing}"


def test_stage_2_8_does_not_add_reward_training_or_model_code() -> None:
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
    ]
    offenders: dict[str, list[str]] = {}
    for path in (ROOT / "src" / "marl_topology").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        hits = [term for term in banned_terms if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"Stage 2.8 added forbidden reward/training/model code: {offenders}"
