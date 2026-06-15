from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage4_2_doc_exists_and_records_boundary() -> None:
    text = _read_doc("STAGE4_2_PBFT_THREE_PHASE_RELIABILITY.md")

    required = [
        "Stage 4.2 PBFT Three-Phase Reliability Record",
        "stage4_pbft_three_phase_closed_form_v0",
        "PBFTThreePhaseConfig",
        "PBFTThreePhaseReliabilityRecord",
        "evaluate_pbft_three_phase_reliability",
        "pre_prepare",
        "prepare",
        "commit",
        "consensus_success_probability",
        "no Stage 3 adapter",
        "no reward",
        "no v5 code migration",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 4.2 doc missing terms: {missing}"


def test_protocol_and_metric_contracts_record_stage4_2_status() -> None:
    protocol = _read_doc("PROTOCOL_CONTRACT.md")
    metric = _read_doc("METRIC_CONTRACT.md")

    protocol_required = [
        "Stage 4.2 PBFT Three-Phase Reliability Record",
        "stage4_pbft_three_phase_closed_form_v0",
        "PBFTThreePhaseConfig",
        "evaluate_pbft_three_phase_reliability",
        "n >= 3f + 1",
        "2f + 1",
        "2f",
        "does not consume Stage 3 communication records",
    ]
    metric_required = [
        "Stage 4.2 PBFT Reliability Mapping",
        "consensus_success_probability",
        "does not add metric names",
        "record diagnostics",
    ]
    missing_protocol = [item for item in protocol_required if item not in protocol]
    missing_metric = [item for item in metric_required if item not in metric]
    assert not missing_protocol, f"PROTOCOL_CONTRACT missing Stage 4.2 terms: {missing_protocol}"
    assert not missing_metric, f"METRIC_CONTRACT missing Stage 4.2 terms: {missing_metric}"


def test_stage4_2_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage4_pbft_three_phase_reliability_record.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage4_pbft_three_phase_reliability_record"
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
    assert "Stage 3 network delivery" in negative_text
    assert "reward" in negative_text
    assert "Monte Carlo" in negative_text
    assert "v5" in negative_text
    assert "training" in negative_text


def test_project_state_marks_stage4_2_complete_and_waits_for_owner() -> None:
    text = _read_doc("PROJECT_STATE.md")

    required = [
        "stage_4_2_pbft_three_phase_reliability_record",
        "stage4_pbft_three_phase_reliability_record_gate",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 4.2 state: {missing}"


def test_stage4_2_source_avoids_adapter_reward_training_and_v5_routes() -> None:
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
        "network_delivery_probability",
        "p2p_latency_s",
        "network_latency_s",
        "network_energy_j",
    ]
    hits = [term for term in banned_terms if term in source]

    assert not hits, f"Stage 4.2 source uses forbidden routes: {hits}"
