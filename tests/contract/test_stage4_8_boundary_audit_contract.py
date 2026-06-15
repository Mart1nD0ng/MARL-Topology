from pathlib import Path

import yaml

from marl_topology.evaluation import (
    STAGE4_8_BOUNDARY_AUDIT_STAGE_ID,
    STAGE4_8_REQUIRED_CASES,
    Stage48BoundaryAuditConfig,
    Stage48BoundaryAuditRow,
    build_stage4_8_boundary_audit_report,
)
from marl_topology.link import LinkTransmissionRecord, RequiredTransmissionTimeResult
from marl_topology.network import NetworkCommunicationRecord


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage4_8_doc_records_boundary_audit() -> None:
    text = _read_doc("STAGE4_8_COMMUNICATION_CONSENSUS_BOUNDARY_AUDIT.md")

    required = [
        "Stage 4.8 Communication/Consensus Boundary Audit",
        "stage_4_8_communication_consensus_boundary_audit",
        "build_stage4_8_boundary_audit_report",
        "network_scheduled_latency_s",
        "network_successful_delivery_latency_s",
        "required_transmission_time_capped",
        "urlcc_finite_blocklength_v1",
        "pbft_expected_initiator_mean_field_v1",
        "full graph remains a baseline",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 4.8 doc missing terms: {missing}"

    for case_name in STAGE4_8_REQUIRED_CASES:
        assert case_name in text


def test_contract_docs_record_stage4_8_semantics() -> None:
    link = _read_doc("LINK_TRANSMISSION_CONTRACT.md")
    network = _read_doc("NETWORK_LAYER_CONTRACT.md")
    protocol = _read_doc("PROTOCOL_CONTRACT.md")
    metric = _read_doc("METRIC_CONTRACT.md")
    plan = _read_doc("STAGE4_PBFT_APPLICATION_CONSENSUS_PLAN.md")

    link_required = [
        "required_reliability_met",
        "required_transmission_time_capped",
        "success_probability_at_required_time",
        "required_time_search_max_s",
        "An unreachable target must not be silently reported",
    ]
    network_required = [
        "network_scheduled_latency_s",
        "network_successful_delivery_latency_s",
        "compatibility alias",
        "failed scheduled messages",
    ]
    protocol_required = [
        "Stage 4.8 Communication/Consensus Boundary Audit",
        "network_scheduled_latency_s",
        "Expected-initiator PBFT remains the consensus reliability model",
    ]
    metric_required = [
        "Stage 4.8 Boundary Audit",
        "does not add metric names",
        "topology_diagnostics",
    ]
    plan_required = [
        "Stage 4.8: Communication/Consensus Boundary Audit",
        "non-saturated",
        "failed scheduled messages",
    ]

    assert not [item for item in link_required if item not in link]
    assert not [item for item in network_required if item not in network]
    assert not [item for item in protocol_required if item not in protocol]
    assert not [item for item in metric_required if item not in metric]
    assert not [item for item in plan_required if item not in plan]


def test_stage4_8_public_interfaces_and_record_fields_exist() -> None:
    report = build_stage4_8_boundary_audit_report()

    assert STAGE4_8_BOUNDARY_AUDIT_STAGE_ID == (
        "stage_4_8_communication_consensus_boundary_audit"
    )
    assert Stage48BoundaryAuditConfig
    assert Stage48BoundaryAuditRow
    assert report["stage"] == STAGE4_8_BOUNDARY_AUDIT_STAGE_ID

    for field_name in [
        "required_reliability_met",
        "required_transmission_time_capped",
        "success_probability_at_required_time",
        "required_time_search_max_s",
    ]:
        assert field_name in LinkTransmissionRecord.__annotations__
        assert field_name in RequiredTransmissionTimeResult.__annotations__

    for field_name in [
        "network_scheduled_latency_s",
        "network_successful_delivery_latency_s",
    ]:
        assert field_name in NetworkCommunicationRecord.__annotations__


def test_stage4_8_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage4_8_boundary_audit.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage4_8_boundary_audit"
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
    assert "scheduled latency" in negative_text
    assert "inverse reliability" in negative_text
    assert "full graph" in negative_text
    assert "Monte Carlo" in negative_text


def test_project_state_marks_stage4_8_complete_and_waits_for_owner() -> None:
    text = _read_doc("PROJECT_STATE.md")

    required = [
        "post_stage_4_8_awaiting_owner_decision",
        "stage_4_8_communication_consensus_boundary_audit",
        "stage4_8_boundary_audit_gate",
        "recommended_next_task: stage_5_0_reward_objective_contract_freeze",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 4.8 state: {missing}"


def test_stage4_8_replay_script_is_print_only() -> None:
    text = (
        ROOT / "scripts" / "replay" / "stage4_8_boundary_audit_report.py"
    ).read_text(encoding="utf-8")

    assert "build_stage4_8_boundary_audit_report" in text
    assert "print(" in text
    banned_terms = ["to_csv", "result_save", "pickle", "torch.save", "open("]
    hits = [term for term in banned_terms if term in text]
    assert not hits, f"Stage 4.8 replay script writes outputs: {hits}"


def test_stage4_8_source_does_not_add_forbidden_routes() -> None:
    scan_paths = [
        ROOT / "src" / "marl_topology" / "evaluation" / "stage4_boundary_audit.py",
        ROOT / "scripts" / "replay" / "stage4_8_boundary_audit_report.py",
    ]
    banned_terms = [
        "D:\\PhD_works\\v5",
        "import v5",
        "from v5",
        "P_eff",
        "train_loop",
        "optimizer",
        "backward(",
        "class Actor(",
        "class Critic(",
        "COMA",
        "MAPPO",
        "def reward",
        "class Reward",
        "Monte Carlo",
        "from random",
        "random.",
        "itertools.combinations",
    ]
    offenders: dict[str, list[str]] = {}
    for path in scan_paths:
        text = path.read_text(encoding="utf-8")
        hits = [term for term in banned_terms if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"Stage 4.8 added forbidden routes: {offenders}"
