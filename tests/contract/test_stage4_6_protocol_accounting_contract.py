from pathlib import Path

import yaml

from marl_topology.protocol import (
    PBFT_PROTOCOL_ACCOUNTING_MODEL_ID,
    PBFTPhaseAccountingRecord,
    PBFTProtocolAccountingRecord,
    account_pbft_protocol_latency_energy,
)


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage4_6_doc_records_protocol_accounting_boundary() -> None:
    text = _read_doc("STAGE4_6_PROTOCOL_LATENCY_ENERGY_ACCOUNTING.md")

    required = [
        "Stage 4.6 Protocol Latency And Energy Accounting Review",
        "stage4_pbft_protocol_accounting_v1",
        "account_pbft_protocol_latency_energy",
        "phase_max_clipped_to_budget",
        "scheduled_attempt_energy_sum",
        "does not export `consensus_success_probability`",
        "does not implement reward",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 4.6 doc missing terms: {missing}"


def test_protocol_metric_and_stage4_plan_record_stage4_6() -> None:
    protocol = _read_doc("PROTOCOL_CONTRACT.md")
    metric = _read_doc("METRIC_CONTRACT.md")
    plan = _read_doc("STAGE4_PBFT_APPLICATION_CONSENSUS_PLAN.md")
    stage45 = _read_doc("STAGE4_5_BASELINE_ORACLE_REVIEW.md")

    protocol_required = [
        "Stage 4.6 Protocol Latency And Energy Accounting Review",
        "stage4_pbft_protocol_accounting_v1",
        "protocol latency and energy are accounting outputs",
    ]
    metric_required = [
        "Stage 4.6 Protocol Latency And Energy Accounting",
        "does not add metric names",
        "`latency`",
        "`energy`",
    ]
    plan_required = [
        "Stage 4.6: Protocol Latency And Energy Accounting Review",
        "Implementation status",
        "not consensus reliability",
    ]
    stage45_required = [
        "Stage 4.6 protocol accounting",
        "stage4_6_sum_phase_max_clipped_to_budget",
    ]
    missing_protocol = [item for item in protocol_required if item not in protocol]
    missing_metric = [item for item in metric_required if item not in metric]
    missing_plan = [item for item in plan_required if item not in plan]
    missing_stage45 = [item for item in stage45_required if item not in stage45]

    assert not missing_protocol, f"PROTOCOL_CONTRACT missing Stage 4.6 terms: {missing_protocol}"
    assert not missing_metric, f"METRIC_CONTRACT missing Stage 4.6 terms: {missing_metric}"
    assert not missing_plan, f"Stage 4 plan missing Stage 4.6 terms: {missing_plan}"
    assert not missing_stage45, f"Stage 4.5 doc missing Stage 4.6 terms: {missing_stage45}"


def test_stage4_6_public_interfaces_exist() -> None:
    assert PBFT_PROTOCOL_ACCOUNTING_MODEL_ID == "stage4_pbft_protocol_accounting_v1"
    assert PBFTPhaseAccountingRecord
    assert PBFTProtocolAccountingRecord
    assert account_pbft_protocol_latency_energy


def test_stage4_6_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage4_6_protocol_accounting_review.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage4_6_protocol_accounting_review"
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
    assert "consensus probability" in negative_text
    assert "reward" in negative_text
    assert "v5" in negative_text


def test_project_state_marks_stage4_6_complete_and_waits_for_owner() -> None:
    text = _read_doc("PROJECT_STATE.md")

    required = [
        "stage_4_6_protocol_latency_energy_accounting_review",
        "stage4_6_protocol_accounting_gate",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 4.6 state: {missing}"


def test_stage4_6_source_does_not_add_model_training_reward_or_v5_routes() -> None:
    scan_paths = [
        ROOT / "src" / "marl_topology" / "protocol" / "pbft_accounting.py",
        ROOT / "src" / "marl_topology" / "evaluation" / "stage4_baseline_oracle_review.py",
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
    ]
    offenders: dict[str, list[str]] = {}
    for path in scan_paths:
        text = path.read_text(encoding="utf-8")
        hits = [term for term in banned_terms if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"Stage 4.6 added forbidden routes: {offenders}"
