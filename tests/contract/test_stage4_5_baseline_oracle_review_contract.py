from pathlib import Path

import yaml

from marl_topology.evaluation import (
    STAGE4_5_REVIEW_STAGE_ID,
    Stage45ReviewConfig,
    Stage45TopologyReviewRow,
    build_stage4_5_baseline_oracle_review,
)


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage4_5_doc_records_boundary_and_outputs() -> None:
    text = _read_doc("STAGE4_5_BASELINE_ORACLE_REVIEW.md")

    required = [
        "Stage 4.5 Baseline And Oracle-Candidate Review",
        "stage_4_5_baseline_and_oracle_review",
        "build_stage4_5_baseline_oracle_review",
        "empty",
        "sparse_star",
        "sparse_chain",
        "full",
        "Full graph is marked as `is_full_graph_baseline = true`",
        "is_deployment_actor_input",
        "not a reward",
        "not a deployment actor input",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 4.5 doc missing terms: {missing}"


def test_protocol_and_metric_contracts_record_stage4_5() -> None:
    protocol = _read_doc("PROTOCOL_CONTRACT.md")
    metric = _read_doc("METRIC_CONTRACT.md")
    plan = _read_doc("STAGE4_PBFT_APPLICATION_CONSENSUS_PLAN.md")

    protocol_required = [
        "Stage 4.5 Baseline And Oracle-Candidate Review",
        "stage_4_5_baseline_and_oracle_review",
        "full graph is a baseline and not an oracle-candidate",
        "failed baselines do not prove infeasibility",
    ]
    metric_required = [
        "Stage 4.5 Baseline And Oracle-Candidate Review",
        "does not add metric names",
        "consensus_success_probability",
        "topology_diagnostics",
    ]
    plan_required = [
        "Stage 4.5: Baseline And Oracle-Candidate Review",
        "Implementation status",
        "oracle-candidate labels are not deployment actor inputs",
    ]
    missing_protocol = [item for item in protocol_required if item not in protocol]
    missing_metric = [item for item in metric_required if item not in metric]
    missing_plan = [item for item in plan_required if item not in plan]

    assert not missing_protocol, f"PROTOCOL_CONTRACT missing Stage 4.5 terms: {missing_protocol}"
    assert not missing_metric, f"METRIC_CONTRACT missing Stage 4.5 terms: {missing_metric}"
    assert not missing_plan, f"Stage 4 plan missing Stage 4.5 terms: {missing_plan}"


def test_stage4_5_public_interfaces_exist() -> None:
    report = build_stage4_5_baseline_oracle_review()

    assert Stage45ReviewConfig
    assert Stage45TopologyReviewRow
    assert report["stage"] == STAGE4_5_REVIEW_STAGE_ID


def test_stage4_5_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage4_5_baseline_oracle_review.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage4_5_baseline_oracle_review"
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
    assert "full graph is labelled oracle" in negative_text
    assert "proof of infeasibility" in negative_text
    assert "reward" in negative_text
    assert "v5" in negative_text


def test_project_state_marks_stage4_5_complete_and_waits_for_owner() -> None:
    text = _read_doc("PROJECT_STATE.md")

    required = [
        "stage_4_5_baseline_and_oracle_review",
        "stage4_5_baseline_oracle_review_gate",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 4.5 state: {missing}"


def test_stage4_5_replay_script_is_print_only() -> None:
    text = (ROOT / "scripts" / "replay" / "stage4_5_baseline_oracle_review.py").read_text(
        encoding="utf-8"
    )

    assert "build_stage4_5_baseline_oracle_review" in text
    assert "print(" in text
    banned_terms = ["to_csv", "result_save", "pickle", "torch.save", "open("]
    hits = [term for term in banned_terms if term in text]
    assert not hits, f"Stage 4.5 replay script writes outputs: {hits}"


def test_stage4_5_source_does_not_add_reward_training_actor_or_v5_routes() -> None:
    scan_paths = [
        ROOT / "src" / "marl_topology" / "evaluation" / "stage4_baseline_oracle_review.py",
        ROOT / "scripts" / "replay" / "stage4_5_baseline_oracle_review.py",
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

    assert not offenders, f"Stage 4.5 added forbidden routes: {offenders}"
