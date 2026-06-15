from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_dec_pomdp_contract_records_stage_2_2_decentralized_baselines() -> None:
    text = (ROOT / "docs" / "DEC_POMDP_CONTRACT.md").read_text(encoding="utf-8")

    required = [
        "Stage 2.2 Decentralized Non-Learning Baselines",
        "ActorObservation",
        "EdgeActionDecision",
        "no_edges",
        "all_local_edges",
        "reliability_threshold",
        "top_k_reliability",
        "local_random",
        "not deployment defaults",
        "not an oracle",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"DEC_POMDP_CONTRACT missing Stage 2.2 baseline terms: {missing}"


def test_project_state_marks_stage_2_2_complete_and_next_env_wrapper() -> None:
    text = (ROOT / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")

    required = [
        "stage_2_2_decentralized_non_learning_baselines",
        "fixed_threshold_is_baseline_gate",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 2.2 state: {missing}"


def test_decentralized_baseline_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "review_decentralized_baselines.yaml"
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
    assert "L006_fixed_threshold_deployment_risk" in task["relevant_lessons"]
    assert "L009_dec_pomdp_boundary_before_actor" in task["relevant_lessons"]


def test_no_decentralized_baseline_model_or_training_code_added() -> None:
    offenders: dict[str, list[str]] = {}
    banned_terms = [
        "class Actor(",
        "class Critic(",
        "COMA",
        "MAPPO",
        "optimizer",
        "train_loop",
    ]
    for path in (ROOT / "src" / "marl_topology").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        hits = [term for term in banned_terms if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"Stage 2.2 added forbidden model/training code: {offenders}"
