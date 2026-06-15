from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_replay_dataset_contract_doc_exists_and_records_boundaries() -> None:
    text = (ROOT / "docs" / "REPLAY_DATASET_CONTRACT.md").read_text(encoding="utf-8")

    required = [
        "Stage 2.6",
        "Replay Dataset Column Contract",
        "Deployment Actor Input",
        "Centralized Training Only",
        "Evaluation Only",
        "reward",
        "Oracle labels",
        "Unknown columns fail validation",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"REPLAY_DATASET_CONTRACT missing terms: {missing}"


def test_dec_pomdp_contract_records_stage_2_6_replay_boundary() -> None:
    text = (ROOT / "docs" / "DEC_POMDP_CONTRACT.md").read_text(encoding="utf-8")

    required = [
        "Stage 2.6 Replay Dataset Column Boundary",
        "Deployment actor input columns",
        "Centralized training-only columns",
        "Evaluation-only columns",
        "src/marl_topology/data/replay_schema.py",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"DEC_POMDP_CONTRACT missing Stage 2.6 terms: {missing}"


def test_project_state_marks_stage_2_6_complete_and_waits_for_owner() -> None:
    text = (ROOT / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")

    required = [
        "stage_2_6_replay_dataset_column_contract",
        "replay_dataset_column_gate",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 2.6 state: {missing}"


def test_replay_dataset_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "replay_dataset_column_contract.yaml"
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
    assert "global topology" in negative_text
    assert "oracle" in negative_text
    assert "reward" in negative_text


def test_stage_2_6_does_not_add_writer_training_reward_or_model_code() -> None:
    banned_terms = [
        "class Actor(",
        "class Critic(",
        "COMA",
        "MAPPO",
        "optimizer",
        "train_loop",
        "open(",
        "write_text",
        "to_csv",
        "P_eff_soft",
        "P_eff_hard",
        "hard_eval",
        "soft_train",
    ]
    offenders: dict[str, list[str]] = {}
    later_manifest_writer_files = {
        Path("src/marl_topology/training/mappo/stage25_pilot.py"),
        Path("src/marl_topology/training/mappo/stage28_repaired_critic_pilot.py"),
    }
    for path in (ROOT / "src" / "marl_topology").rglob("*.py"):
        relative = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        hits = []
        for term in banned_terms:
            if term not in text:
                continue
            if relative in later_manifest_writer_files and term in {"open(", "write_text"}:
                continue
            hits.append(term)
        if hits:
            offenders[str(relative)] = hits

    assert not offenders, f"Stage 2.6 added forbidden writer/model/training code: {offenders}"
