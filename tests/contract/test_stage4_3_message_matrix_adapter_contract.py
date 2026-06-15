from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage4_3_doc_exists_and_records_boundary() -> None:
    text = _read_doc("STAGE4_3_STAGE3_MESSAGE_MATRIX_ADAPTER.md")

    required = [
        "Stage 4.3 Stage 3 Message-Matrix Adapter",
        "stage4_stage3_network_to_pbft_matrix_v1",
        "PBFTPhaseBudgets",
        "PBFTMessageMatrices",
        "build_pbft_message_matrices_from_network_records",
        "evaluate_pbft_reliability_from_network_records",
        "network_delivery_probability",
        "network_latency_s <= phase_budget_s",
        "exports_consensus_metric = false",
        "does not expose `consensus_success_probability`",
        "does not",
        "migrate v5 code",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 4.3 doc missing terms: {missing}"


def test_protocol_and_metric_contracts_record_stage4_3_boundary() -> None:
    protocol = _read_doc("PROTOCOL_CONTRACT.md")
    metric = _read_doc("METRIC_CONTRACT.md")

    protocol_required = [
        "Stage 4.3 Stage 3 Message-Matrix Adapter",
        "stage4_stage3_network_to_pbft_matrix_v1",
        "PBFTPhaseBudgets",
        "build_pbft_message_matrices_from_network_records",
        "network_delivery_probability",
        "network_latency_s <= phase_budget_s",
        "does not export a consensus metric",
        "does not rename `network_delivery_probability`",
    ]
    metric_required = [
        "Stage 4.3 Message-Matrix Adapter Boundary",
        "does not add metric names",
        "`network_delivery_probability` remains a Stage 3 communication record field",
        "must not export it as `consensus_success_probability`",
    ]
    missing_protocol = [item for item in protocol_required if item not in protocol]
    missing_metric = [item for item in metric_required if item not in metric]
    assert not missing_protocol, f"PROTOCOL_CONTRACT missing Stage 4.3 terms: {missing_protocol}"
    assert not missing_metric, f"METRIC_CONTRACT missing Stage 4.3 terms: {missing_metric}"


def test_stage4_3_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage4_stage3_message_matrix_adapter.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage4_stage3_message_matrix_adapter"
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
    assert "network_delivery_probability" in negative_text
    assert "consensus_success_probability" in negative_text
    assert "reward" in negative_text
    assert "v5" in negative_text


def test_project_state_marks_stage4_3_complete_and_waits_for_owner() -> None:
    text = _read_doc("PROJECT_STATE.md")

    required = [
        "stage_4_3_stage3_message_matrix_adapter",
        "stage4_stage3_message_matrix_adapter_gate",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 4.3 state: {missing}"


def test_stage4_3_source_avoids_reward_training_and_v5_routes() -> None:
    source = (
        ROOT / "src" / "marl_topology" / "protocol" / "message_matrix_adapter.py"
    ).read_text(encoding="utf-8")
    banned_terms = [
        "import v5",
        "from v5",
        "torch",
        "class Actor(",
        "class Critic(",
        "COMA",
        "MAPPO",
        "optimizer",
        "backward(",
        "train_loop",
        "torch.save",
        "P_eff",
        "reward",
    ]
    hits = [term for term in banned_terms if term in source]

    assert not hits, f"Stage 4.3 source uses forbidden routes: {hits}"
