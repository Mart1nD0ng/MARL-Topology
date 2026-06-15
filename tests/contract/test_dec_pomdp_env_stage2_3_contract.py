from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_dec_pomdp_contract_records_stage_2_3_env_wrapper() -> None:
    text = (ROOT / "docs" / "DEC_POMDP_CONTRACT.md").read_text(encoding="utf-8")

    required = [
        "Stage 2.3 Minimal Dec-POMDP Env Wrapper",
        "reset()",
        "step(local_decisions)",
        "ActorObservation",
        "EdgeActionDecision",
        "TopologyEvaluation",
        "does not compute reward",
        "does not train",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"DEC_POMDP_CONTRACT missing Stage 2.3 env terms: {missing}"


def test_project_state_marks_stage_2_3_complete() -> None:
    text = (ROOT / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")

    required = [
        "stage_2_3_minimal_dec_pomdp_env_wrapper",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 2.3 state: {missing}"


def test_dec_pomdp_env_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "review_dec_pomdp_env_wrapper.yaml"
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


def test_stage_2_3_does_not_add_model_training_or_reward_code() -> None:
    banned_terms = [
        "class Actor(",
        "class Critic(",
        "COMA",
        "MAPPO",
        "optimizer",
        "train_loop",
        "reward =",
        "reward:",
    ]
    offenders: dict[str, list[str]] = {}
    for path in (ROOT / "src" / "marl_topology").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        hits = [term for term in banned_terms if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"Stage 2.3 added forbidden model/training/reward code: {offenders}"
