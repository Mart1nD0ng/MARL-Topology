from pathlib import Path

import yaml

from marl_topology.protocol import (
    PBFT_EXPECTED_INITIATOR_MODEL_ID,
    PBFTExpectedInitiatorConfig,
    PBFTExpectedInitiatorReliabilityRecord,
    evaluate_expected_initiator_pbft_reliability,
    evaluate_pbft_given_primary,
)


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage4_4_docs_record_expected_initiator_model() -> None:
    reliability_doc = _read_doc("STAGE4_2_PBFT_THREE_PHASE_RELIABILITY.md")
    protocol = _read_doc("PROTOCOL_CONTRACT.md")
    metric = _read_doc("METRIC_CONTRACT.md")

    required_reliability = [
        "Stage 4.4 adds topology-level expected initiator reliability",
        "pbft_expected_initiator_mean_field_v1",
        "R_consensus(G, s) = (1 / |V|) * sum_p Psi_p(G, s)",
        "primary_distribution = uniform",
        "view_change_mode = deferred",
        "not a strict Byzantine adversary model",
    ]
    required_protocol = [
        "Stage 4.4 Expected-Initiator PBFT Reliability",
        "evaluate_expected_initiator_pbft_reliability",
        "A fixed primary is only an internal helper",
        "does not implement view-change",
    ]
    required_metric = [
        "Stage 4.2 / Stage 4.4 PBFT Reliability Mapping",
        "uniform average of per-primary reliability values",
        "does not add metric names",
    ]
    missing_reliability = [item for item in required_reliability if item not in reliability_doc]
    missing_protocol = [item for item in required_protocol if item not in protocol]
    missing_metric = [item for item in required_metric if item not in metric]

    assert not missing_reliability, f"Stage 4.4 doc missing terms: {missing_reliability}"
    assert not missing_protocol, f"PROTOCOL_CONTRACT missing Stage 4.4 terms: {missing_protocol}"
    assert not missing_metric, f"METRIC_CONTRACT missing Stage 4.4 terms: {missing_metric}"


def test_stage4_4_public_interfaces_exist() -> None:
    assert PBFTExpectedInitiatorConfig
    assert PBFTExpectedInitiatorReliabilityRecord
    assert evaluate_pbft_given_primary
    assert evaluate_expected_initiator_pbft_reliability
    assert PBFT_EXPECTED_INITIATOR_MODEL_ID == "pbft_expected_initiator_mean_field_v1"


def test_stage4_4_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage4_4_expected_initiator_pbft_reliability.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage4_4_expected_initiator_pbft_reliability"
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
    assert "fixed global primary" in negative_text
    assert "strict Byzantine adversary" in negative_text
    assert "Monte Carlo" in negative_text
    assert "v5" in negative_text


def test_project_state_marks_stage3_6_and_stage4_4_complete() -> None:
    text = _read_doc("PROJECT_STATE.md")

    required = [
        "stage_3_6_urlcc_finite_blocklength_link_reliability",
        "stage_4_4_expected_initiator_pbft_reliability",
        "stage3_6_urlcc_finite_blocklength_gate",
        "stage4_4_expected_initiator_pbft_gate",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 3.6 / 4.4 state: {missing}"


def test_stage4_4_source_avoids_forbidden_routes() -> None:
    source = (ROOT / "src" / "marl_topology" / "protocol" / "pbft_reliability.py").read_text(
        encoding="utf-8"
    )
    banned_terms = [
        "itertools",
        "combinations",
        "permutations",
        "monte_carlo",
        "Monte Carlo",
        "random",
        "sample",
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
    ]
    hits = [term for term in banned_terms if term in source]

    assert not hits, f"Stage 4.4 source uses forbidden routes: {hits}"
