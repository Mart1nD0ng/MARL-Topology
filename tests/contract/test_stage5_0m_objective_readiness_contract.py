import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_doc(name: str) -> str:
    return _read(ROOT / "docs" / name)


def test_stage5_0m_documentation_exists_and_records_plan_only_decision() -> None:
    text = _read_doc("STAGE5_0M_OBJECTIVE_READINESS_REVIEW.md")

    required = [
        "Stage 5.0m",
        "tau_requirement_min = 0.9",
        "Stage 5.1 - reward implementation plan without code",
        "stage5_1_plan_only_allowed = true",
        "reward_code_allowed = false",
        "training_allowed = false",
        "final_tau_selected = false",
        "Stage 3-backed sweep coverage",
        "does not migrate v5 code",
        "full graph as an oracle",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 5.0m doc missing terms: {missing}"


def test_stage5_0m_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage5_0m_objective_readiness_review.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage5_0m_objective_readiness_review"
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
    assert "Stage 5.1 is treated as permission to implement reward code" in negative_text
    assert "final tau lower than 0.9" in negative_text
    assert "reward implementation" in negative_text
    assert "full graph" in negative_text
    assert "strict Byzantine model" in negative_text


def test_stage5_0m_replay_script_prints_readiness_report() -> None:
    script = ROOT / "scripts" / "replay" / "stage5_0m_objective_readiness_review.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert '"stage": "stage_5_0m_objective_readiness_review_before_reward_implementation"' in completed.stdout
    assert '"tau_requirement_min": 0.9' in completed.stdout
    assert '"stage5_1_plan_only_allowed": true' in completed.stdout
    assert '"recommended_next_task": "stage_5_1_reward_implementation_plan_without_code"' in completed.stdout
    assert '"reward_code_allowed": false' in completed.stdout
    assert '"training_allowed": false' in completed.stdout
    assert '"final_tau_selected": false' in completed.stdout
    assert '"all_readiness_gates_passed": true' in completed.stdout


def test_stage5_0m_project_state_marks_ready_for_stage5_1_owner_decision() -> None:
    text = _read_doc("PROJECT_STATE.md")

    required = [
        "post_stage_5_0m_ready_for_stage_5_1_owner_decision",
        "post_stage_5_0l_awaiting_owner_decision",
        "stage_5_0m_objective_readiness_review_before_reward_implementation",
        "stage5_0m_objective_readiness_review_gate",
        "recommended_next_task: stage_5_1_reward_implementation_plan_without_code",
        "stage_5_1_reward_implementation_plan_without_code",
        "reward_implementation",
        "training_runs",
        "v5_code_migration",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 5.0m state: {missing}"


def test_stage5_0m_contract_notes_do_not_authorize_reward_code() -> None:
    objective = _read_doc("OBJECTIVE_CONTRACT.md")
    surrogate = _read_doc("REWARD_SURROGATE_CONTRACT.md")

    assert "Stage 5.0m" in objective
    assert "reward implementation plan without code" in objective
    assert "Stage 5.0m" in surrogate
    assert "Stage 5.1 may design a reward implementation plan" in surrogate
    assert "does not permit reward implementation" in surrogate


def test_stage5_0m_source_avoids_sampling_model_code_v5_and_legacy_metrics() -> None:
    source = _read(
        ROOT
        / "src"
        / "marl_topology"
        / "evaluation"
        / "objective_readiness_review.py"
    )
    banned_terms = [
        "random",
        "sample",
        "Monte Carlo",
        "combinations(",
        "class Actor(",
        "class Critic(",
        "COMA",
        "MAPPO",
        "optimizer",
        "backward(",
        "train_loop",
        "def reward",
        "def compute_reward",
        "class Reward",
        "P_eff_soft",
        "P_eff_hard",
        "hard_eval",
        "soft_train",
        "D:\\PhD_works\\v5",
    ]
    hits = [term for term in banned_terms if term in source]
    assert not hits, f"Stage 5.0m source added forbidden terms: {hits}"
