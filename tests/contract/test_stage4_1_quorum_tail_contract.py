from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage4_1_quorum_tail_doc_exists_and_records_boundary() -> None:
    text = _read_doc("STAGE4_1_QUORUM_TAIL_UTILITY.md")

    required = [
        "Stage 4.1 Heterogeneous Quorum-Tail Utility",
        "stage4_heterogeneous_quorum_tail_v1",
        "heterogeneous_quorum_tail",
        "conservative_quorum_tail",
        "does not enumerate success subsets",
        "not a new project metric",
        "not a PBFT three-phase implementation",
        "does not import v5 code",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 4.1 quorum-tail doc missing terms: {missing}"


def test_protocol_contract_records_stage4_1_implementation_status() -> None:
    text = _read_doc("PROTOCOL_CONTRACT.md")

    required = [
        "Stage 4.1 Heterogeneous Quorum-Tail Utility",
        "stage4_heterogeneous_quorum_tail_v1",
        "src/marl_topology/protocol/quorum_tail.py",
        "heterogeneous_quorum_tail",
        "conservative_quorum_tail",
        "not a PBFT three-phase implementation",
        "does not emit `consensus_success_probability` as a project metric",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROTOCOL_CONTRACT missing Stage 4.1 terms: {missing}"


def test_stage4_1_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage4_heterogeneous_quorum_tail_utility.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage4_heterogeneous_quorum_tail_utility"
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
    assert "subsets" in negative_text
    assert "Monte Carlo" in negative_text
    assert "v5" in negative_text
    assert "PBFT pre-prepare prepare commit cascade" in negative_text


def test_project_state_marks_stage4_1_complete_and_waits_for_owner() -> None:
    text = _read_doc("PROJECT_STATE.md")

    required = [
        "stage_4_1_heterogeneous_quorum_tail_utility",
        "stage4_heterogeneous_quorum_tail_utility_gate",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 4.1 state: {missing}"


def test_stage4_1_does_not_add_reward_training_or_v5_routes() -> None:
    paths = [
        ROOT / "src" / "marl_topology" / "protocol" / "quorum_tail.py",
    ]
    banned_terms = [
        "class Actor(",
        "class Critic(",
        "COMA",
        "MAPPO",
        "optimizer",
        "backward(",
        "train_loop",
        "torch.save",
        "import v5",
        "from v5",
        "P_eff",
    ]
    offenders: dict[str, list[str]] = {}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        hits = [term for term in banned_terms if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"Stage 4.1 introduced forbidden routes: {offenders}"
