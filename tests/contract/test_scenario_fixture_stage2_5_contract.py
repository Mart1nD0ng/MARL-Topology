from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_scenario_fixture_contract_doc_exists_and_records_boundaries() -> None:
    text = (ROOT / "docs" / "SCENARIO_FIXTURE_CONTRACT.md").read_text(encoding="utf-8")

    required = [
        "Stage 2.5",
        "Scenario Fixture Contract",
        "demo_stage2",
        "sparse_chain_stage2",
        "quorum_blocked_stage2",
        "stage2_deterministic_distance",
        "Full graph is a baseline, not an oracle",
        "No fixture may introduce a new metric",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"SCENARIO_FIXTURE_CONTRACT missing terms: {missing}"


def test_project_state_marks_stage_2_5_complete_and_waits_for_owner() -> None:
    text = (ROOT / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")

    required = [
        "stage_2_5_scenario_fixture_contract",
        "scenario_fixture_contract_gate",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 2.5 state: {missing}"


def test_scenario_fixture_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "scenario_fixture_contract.yaml"
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
    assert "stochastic physics" in negative_text
    assert "full graph" in negative_text
    assert "oracle" in negative_text


def test_scenario_fixture_report_script_is_replay_only() -> None:
    text = (ROOT / "scripts" / "replay" / "scenario_fixture_report.py").read_text(
        encoding="utf-8"
    )

    assert "build_stage2_scenario_fixture_report" in text
    assert "open(" not in text
    assert "write_text" not in text
    assert "result_save" not in text


def test_stage_2_5_does_not_add_model_training_reward_or_legacy_metric_defaults() -> None:
    banned_terms = [
        "class Actor(",
        "class Critic(",
        "COMA",
        "MAPPO",
        "optimizer",
        "train_loop",
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

    assert not offenders, f"Stage 2.5 added forbidden code or metric defaults: {offenders}"
