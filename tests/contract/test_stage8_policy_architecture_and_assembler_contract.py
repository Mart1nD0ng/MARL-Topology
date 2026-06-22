from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage8_1_architecture_decision_document_exists() -> None:
    text = _read_doc("STAGE8_1_POLICY_ARCHITECTURE_DECISION.md")

    required = [
        "decentralized local outgoing directed edge scorer",
        "Hard activation is produced by the environment-side topology assembler",
        "A1 Local MLP edge scorer",
        "A2 Local GNN edge scorer",
        "A3 Local GRU edge scorer",
        "A4 Local GNN + GRU edge scorer",
        "A5 Local GNN + LSTM edge scorer",
        "A6 Local neighbor-attention / local transformer edge scorer",
        "C1 global pooled MLP critic",
        "C2 centralized GNN critic",
        "C3 centralized GNN critic + edge-delta heads",
        "C4 centralized Graph Transformer critic",
        "COMA is not mainline",
        "future optional ablation",
        "Direct edge-delta critic evidence is preferred",
        "torch neural models",
        "training loop",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 8.1 decision doc missing terms: {missing}"


def test_stage8_2_topology_assembler_contract_document_exists() -> None:
    text = _read_doc("STAGE8_2_TOPOLOGY_ASSEMBLER_CONTRACT.md")

    required = [
        "Deployment Assembler",
        "Training Assembler",
        "Oracle Assembler",
        "The assembler is an action feasibility projection and resource scheduling layer",
        "Full graph remains a baseline, not an oracle",
        "Fixed per-agent top-k remains a baseline",
        "ThresholdAssembler",
        "FixedTopKPerAgentAssembler",
        "RoleAwareAdaptiveBudgetAssembler",
        "ConflictAwareGreedyAssembler",
        "recommended Stage 8 deployment assembler",
        "consensus_success_probability",
        "Stage 4 PBFT reliability result",
        "Gumbel top-k",
        "Plackett-Luce subset sampling",
        "sequential categorical proposal policy",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 8.2 assembler contract missing terms: {missing}"


def test_stage8_source_scan_blocks_model_training_and_legacy_migration() -> None:
    banned_terms = [
        "import tensorflow",
        "from tensorflow",
        "class COMA",
        "def coma",
        "class PPO",
        "class MAPPO",
        "def train_loop",
        "torch.save",
        "checkpoint_path",
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
        for term in banned_terms:
            if term in text:
                hits.append(f"{relative}:{term}")
    assert not torch_hits, f"unexpected direct torch imports outside model/training modules: {torch_hits}"
    assert not hits, f"Stage 8 introduced forbidden source patterns: {hits}"


def test_stage8_project_state_marks_stage8_closed_and_stage9_owner_gated() -> None:
    text = _read_doc("PROJECT_STATE.md")

    required = [
        "post_stage_8_complete_awaiting_owner_decision_for_stage_9",
        "stage_8_policy_architecture_and_topology_assembler_deployment",
        "stage8_policy_architecture_and_assembler_gate",
        "stage8_topology_assembler_gate",
        "stage_9_0_local_mlp_edge_scorer_baseline_without_training_execution",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 8 closure terms: {missing}"


def test_stage8_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage8_policy_architecture_and_assembler.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage8_policy_architecture_and_assembler"
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
    assert "neural model implementation is added" in negative_text
    assert "deployment assembler consumes objective oracle reward" in negative_text
    assert "fixed top-k is presented as final policy" in negative_text
    assert "COMA implementation is added" in negative_text
