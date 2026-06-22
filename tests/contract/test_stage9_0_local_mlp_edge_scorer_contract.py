from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage9_0_document_exists_and_states_no_training() -> None:
    text = _read_doc("STAGE9_0_LOCAL_MLP_EDGE_SCORER_BASELINE.md")

    required = [
        "Stage 9.0 Local MLP Edge Scorer Baseline Scaffold",
        "actor_policy_local_edge_score_output_v1",
        "activate belongs only to legacy local decisions or assembler-selected topology",
        "LocalMLPEdgeScorer",
        "no training execution",
        "no optimizer.step",
        "no checkpoint",
        "no training artifact",
        "no PPO/MAPPO/COMA",
        "no GNN/GRU/LSTM/Transformer",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 9.0 doc missing terms: {missing}"


def test_stage9_0_project_state_marks_stage9_0_complete_and_owner_gated() -> None:
    text = _read_doc("PROJECT_STATE.md")

    required = [
        "post_stage_9_0_local_mlp_edge_scorer_scaffold_awaiting_owner_decision",
        "stage_9_0_local_mlp_edge_scorer_baseline_without_training_execution",
        "stage9_0_active_edge_score_schema_gate",
        "stage9_0_no_training_execution_gate",
        "stage_9_1_supervised_edge_scoring_warm_start_plan_without_training_execution",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 9.0 terms: {missing}"


def test_stage9_0_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage9_0_local_mlp_edge_scorer_baseline.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage9_0_local_mlp_edge_scorer_baseline"
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
    assert "actor output schema still treats activate as active model output" in negative_text
    assert "optimizer.step" in negative_text
    assert "checkpoint" in negative_text
    assert "PPO/MAPPO/COMA" in negative_text


def test_stage9_0_source_scan_allows_only_local_mlp_torch_model() -> None:
    forbidden_terms = [
        "optimizer.step",
        "train_loop",
        "torch.save",
        "torch.load",
        "class PPO",
        "class MAPPO",
        "class COMA",
        "class GNN",
        "class GRU",
        "class LSTM",
        "class Transformer",
        "D:\\PhD_works\\v5",
    ]
    hits: list[str] = []
    torch_hits: list[str] = []
    for path in (ROOT / "src" / "marl_topology").rglob("*.py"):
        relative = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        torch_allowed = relative.parts[:3] in {
            ("src", "marl_topology", "models"),
            ("src", "marl_topology", "training"),
        } or relative.as_posix() == "src/marl_topology/protocol/torch_quorum_tail.py"
        # ^ Phase 1c (spec-driven reconstruction): the Technical-Spec mandates a
        #   differentiable Torch quorum-tail at protocol/torch_quorum_tail.py (Spec S4.4-4.5).
        #   It is a standalone submodule (not imported by protocol/__init__), so the base
        #   protocol package stays Torch-free.
        if ("import torch" in text or "from torch" in text) and not torch_allowed:
            torch_hits.append(relative.as_posix())
        for term in forbidden_terms:
            if term in text:
                hits.append(f"{relative.as_posix()}:{term}")
    assert not torch_hits, f"unexpected direct torch imports outside model/training modules: {torch_hits}"
    assert not hits, f"Stage 9.0 source introduced forbidden patterns: {hits}"
