from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_protocol_timeout_review_doc_exists_and_declares_boundary() -> None:
    text = (ROOT / "docs" / "PROTOCOL_TIMEOUT_REVIEW.md").read_text(encoding="utf-8")

    required = [
        "Stage 2.8 Protocol Timeout Review",
        "stage2_minimal_quorum_graph",
        "does not implement full PBFT",
        "deadline_s = None disables the deadline gate",
        "equality passes",
        "consensus_success_probability is the Stage 2 link/topology probability",
        "not multiplied by timeout or quorum factors",
        "Timeout and quorum are not reward terms",
        "v5 remains a read-only experience library",
        "Deadline equality passes",
        "consensus_success_probability is not timeout-gated in Stage 2",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROTOCOL_TIMEOUT_REVIEW missing terms: {missing}"


def test_protocol_contract_records_stage_2_8_timeout_semantics() -> None:
    text = (ROOT / "docs" / "PROTOCOL_CONTRACT.md").read_text(encoding="utf-8")

    required = [
        "Stage 2.8 Minimal Quorum / Timeout Review",
        "stage2_minimal_quorum_graph",
        "`latency <= deadline_s`",
        "`latency > deadline_s` is a timeout event",
        "Timeout affects binary `consensus_success`",
        "`consensus_success_probability` is not timeout-gated",
        "Failure-reason priority is diagnostic only",
        "No `P_eff`, hard/soft/legacy mode taxonomy",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROTOCOL_CONTRACT missing Stage 2.8 terms: {missing}"


def test_protocol_timeout_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "protocol_timeout_review_stage2_8.yaml"
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
    for expected in ["timeout quorum", "unregistered metric", "P_eff", "full PBFT", "training"]:
        assert expected in negative_text


def test_project_state_marks_protocol_timeout_review_complete() -> None:
    text = (ROOT / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")

    required = [
        "stage_2_8_protocol_timeout_review",
        "protocol_timeout_gate",
        "consensus_protocol_naming_gate",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing protocol timeout state: {missing}"


def test_stage_2_8_protocol_review_does_not_add_forbidden_protocol_code() -> None:
    banned_terms = [
        "P_eff_soft",
        "P_eff_hard",
        "hard_eval",
        "soft_train",
        "timeout_reward",
        "quorum_reward",
        "class PBFT",
        "view_change",
        "byzantine",
        "train_loop",
        "optimizer",
        "backward(",
    ]
    offenders: dict[str, list[str]] = {}
    for path in (ROOT / "src" / "marl_topology" / "protocol").rglob("*.py"):
        if path.name in {
            "quorum_tail.py",
            "pbft_reliability.py",
            "message_matrix_adapter.py",
            "pbft_accounting.py",
        }:
            continue
        text = path.read_text(encoding="utf-8")
        hits = [term for term in banned_terms if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"Stage 2.8 protocol review added forbidden code: {offenders}"
