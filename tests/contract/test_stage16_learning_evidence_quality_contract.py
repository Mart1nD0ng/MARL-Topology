from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage16_document_answers_required_quality_questions() -> None:
    text = _read_doc("STAGE16_LEARNING_EVIDENCE_QUALITY_IMPROVEMENT.md")

    required = [
        "Stage 16",
        "tau_requirement_min=0.9",
        "feasible examples",
        "near-threshold examples",
        "infeasible hard examples",
        "sparse feasible topology",
        "full graph resource-dominated topology",
        "weak-primary / center-primary contrast",
        "interference penalty examples",
        "real multi-step actor-safe sequence",
        "add edge",
        "remove edge",
        "keep edge",
        "delta consensus_success_probability",
        "delta latency",
        "delta energy",
        "delta surrogate diagnostic",
        "current data is not sufficient to continue supervised training",
        "identical local observations with contradictory labels",
        "continue evidence expansion before rerunning Stage 11-15",
        "scale-up training remains blocked",
    ]
    missing = [term for term in required if term not in text]
    assert not missing, f"Stage 16 report missing terms: {missing}"


def test_stage16_project_state_repairs_stage9_to_stage15_history() -> None:
    text = _read_doc("PROJECT_STATE.md")

    required = [
        "current_stage: post_stage_16_complete_awaiting_owner_decision_for_stage_17",
        "post_stage_15_complete_awaiting_owner_decision_for_stage_16",
        "stage_10_supervised_loss_dry_run",
        "stage_11_supervised_mlp_actor_warm_start",
        "stage_12_critic_edge_delta_pretraining",
        "stage_13_local_gnn_edge_scorer_comparison",
        "stage_14_gru_lstm_temporal_actor_ablation",
        "stage_15_controlled_policy_gradient_pilot",
        "stage_16_learning_evidence_quality_improvement",
        "recommended_next_task: stage_17_actor_observability_and_label_disambiguation",
        "owner_decision_required: true",
        "scale-up training remains blocked",
    ]
    missing = [term for term in required if term not in text]
    assert not missing, f"PROJECT_STATE missing Stage 16 repair terms: {missing}"


def test_stage16_harness_tasks_exist_and_are_registered() -> None:
    review_path = ROOT / "harness" / "tasks" / "stage9_to_stage15_model_stack_goal_review.yaml"
    stage16_path = ROOT / "harness" / "tasks" / "stage16_learning_evidence_quality_improvement.yaml"

    review = yaml.safe_load(review_path.read_text(encoding="utf-8"))
    stage16 = yaml.safe_load(stage16_path.read_text(encoding="utf-8"))

    assert review["id"] == "stage9_to_stage15_model_stack_goal_review"
    assert stage16["id"] == "stage16_learning_evidence_quality_improvement"
    assert "Stage 16" in stage16["name"]
    assert any("no scale-up training" in item for item in stage16["negative_checks"])
    assert any("Stage 10-15" in item for item in review["expected_evidence"])


def test_stage16_contract_does_not_add_training_artifact_or_model_files() -> None:
    stage16_source_files = [
        ROOT / "src" / "marl_topology" / "data" / "learning_evidence_stage16.py",
    ]
    forbidden_terms = [
        "optimizer.step",
        ".backward(",
        "torch.save",
        "write_checkpoint",
        "checkpoint_path",
        "training_loop",
        "COMA implementation",
        "Transformer implementation",
        "D:\\PhD_works\\v5",
    ]
    hits = []
    for path in stage16_source_files:
        text = path.read_text(encoding="utf-8")
        for term in forbidden_terms:
            if term in text:
                hits.append(f"{path.relative_to(ROOT)}:{term}")
    assert not hits, f"Stage 16 contract introduced forbidden implementation terms: {hits}"
