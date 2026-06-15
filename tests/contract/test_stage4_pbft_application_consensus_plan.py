from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage3_coupling_review_records_current_dependency_direction() -> None:
    text = _read_doc("STAGE3_LATENCY_ENERGY_RELIABILITY_COUPLING_REVIEW.md")

    required = [
        "urlcc_finite_blocklength_v1",
        "packet_success_probability = 1 - epsilon",
        "deadline_delivery_probability",
        "expected_latency_s = expected_attempts * t_attempt",
        "expected_energy_j = expected_attempts * attempt_energy_j",
        "not `consensus_success_probability`",
        "Stage 4 consumes communication delivery",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 3 coupling review missing terms: {missing}"


def test_stage4_pbft_plan_declares_closed_form_three_phase_variant() -> None:
    text = _read_doc("STAGE4_PBFT_APPLICATION_CONSENSUS_PLAN.md")

    required = [
        "stage4_pbft_three_phase_closed_form_v0",
        "pbft_expected_initiator_mean_field_v1",
        "pre_prepare",
        "prepare",
        "commit",
        "H_ge_k",
        "generating-polynomial",
        "consensus_success_probability",
        "n >= 3f + 1",
        "2f + 1",
        "2f",
        "mean-field approximation",
        "Stage 4.1: Quorum Tail Utility",
        "Stage 4.4: Expected-Initiator PBFT Reliability",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 4 PBFT plan missing terms: {missing}"


def test_stage4_plan_rejects_sampling_subset_enumeration_reward_and_v5_copying() -> None:
    text = _read_doc("STAGE4_PBFT_APPLICATION_CONSENSUS_PLAN.md")

    required = [
        "do not enumerate subsets",
        "do not use Monte Carlo",
        "random sampling",
        "learned prediction",
        "reward terms",
        "import v5 code",
        "network_delivery_probability",
        "full graph",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 4 negative checks missing terms: {missing}"


def test_protocol_and_metric_contracts_record_stage4_boundary() -> None:
    protocol = _read_doc("PROTOCOL_CONTRACT.md")
    metric = _read_doc("METRIC_CONTRACT.md")

    protocol_required = [
        "Stage 4.0 PBFT / Application Consensus Planning",
        "stage4_pbft_three_phase_closed_form_v0",
        "pbft_expected_initiator_mean_field_v1",
        "pre-prepare",
        "prepare",
        "commit",
        "not subset enumeration",
        "not inherit v5 metric aliases",
    ]
    metric_required = [
        "Stage 4.0 PBFT Planning Note",
        "consensus_success_probability",
        "analytic formula source",
        "no sampling or subset enumeration",
        "network delivery",
    ]
    missing_protocol = [item for item in protocol_required if item not in protocol]
    missing_metric = [item for item in metric_required if item not in metric]
    assert not missing_protocol, f"PROTOCOL_CONTRACT missing Stage 4 terms: {missing_protocol}"
    assert not missing_metric, f"METRIC_CONTRACT missing Stage 4 terms: {missing_metric}"


def test_stage4_harness_task_exists_with_negative_checks() -> None:
    path = ROOT / "harness" / "tasks" / "stage4_pbft_application_consensus_plan.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage4_pbft_application_consensus_plan"
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
    assert "Monte Carlo" in negative_text
    assert "subset enumeration" in negative_text
    assert "reward" in negative_text
    assert "v5" in negative_text


def test_project_state_marks_stage4_0_complete_and_waits_for_owner() -> None:
    text = _read_doc("PROJECT_STATE.md")

    required = [
        "stage_4_0_pbft_application_consensus_contract_planning",
        "stage4_pbft_application_consensus_plan_gate",
        "stage_4_1_heterogeneous_quorum_tail_utility",
        "stage_4_pbft_consensus_implementation_without_owner_approval",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 4.0 state: {missing}"


def test_stage4_0_does_not_add_protocol_implementation_code() -> None:
    protocol_source = (ROOT / "src" / "marl_topology" / "protocol").rglob("*.py")
    banned_terms = [
        "stage4_pbft_three_phase_closed_form_v0",
        "H_ge_k",
        "monte_carlo",
        "random.",
        "import v5",
    ]
    offenders: dict[str, list[str]] = {}
    for path in protocol_source:
        if path.name in {"quorum_tail.py", "pbft_reliability.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        hits = [term for term in banned_terms if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"Stage 4.0 added protocol implementation code: {offenders}"
