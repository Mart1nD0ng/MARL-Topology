"""Deployment-layer purity gate (was the Stage 9.0 local-MLP scorer contract; doc lineage retired).

The deployed inference path -- everything under ``src/marl_topology`` EXCEPT the ``models/`` and
``training/`` CTDE subtrees -- must stay decentralized and training-free: no optimizer / train
loop, no checkpoint I/O, no centralized critic/actor or graph/recurrent/PPO/MAPPO/COMA model
classes, no torch imports (save the spec-mandated differentiable quorum tail), and no v5 lineage.
A centralized graph critic + Graph-MAPPO/COMA/PPO + optimizers + checkpoints live LEGALLY under
``models/`` and ``training/`` (Engineering-Plan Phases 7-13, CTDE) and are exempt here -- that is
exactly the CTDE boundary (D1): centralized training, fully decentralized execution. The stale
doc / PROJECT_STATE / harness-task assertions were retired on 2026-06-23 with the multi-stage
process lineage; this real deployment-purity scan survives and now governs the CTDE unlock.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_CTDE_TRAINING_SUBTREES = {
    ("src", "marl_topology", "models"),
    ("src", "marl_topology", "training"),
}


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
        "class Critic(",
        "class Actor(",
        "D:\\PhD_works\\v5",
    ]
    hits: list[str] = []
    torch_hits: list[str] = []
    for path in (ROOT / "src" / "marl_topology").rglob("*.py"):
        relative = path.relative_to(ROOT)
        if relative.parts[:3] in _CTDE_TRAINING_SUBTREES:
            # CTDE training subtrees (centralized critic + MAPPO/COMA/PPO + optimizers +
            # checkpoints) are legal here per Engineering-Plan Phases 7-13 -- not deployed paths.
            continue
        text = path.read_text(encoding="utf-8")
        # The Technical-Spec mandates a differentiable Torch quorum tail at
        # protocol/torch_quorum_tail.py (Spec S4.4-4.5); it is a standalone submodule (not
        # imported by protocol/__init__), so the base protocol package stays Torch-free. Every
        # other deployed path must import no torch at all.
        torch_allowed = relative.as_posix() == "src/marl_topology/protocol/torch_quorum_tail.py"
        if ("import torch" in text or "from torch" in text) and not torch_allowed:
            torch_hits.append(relative.as_posix())
        for term in forbidden_terms:
            if term in text:
                hits.append(f"{relative.as_posix()}:{term}")
    assert not torch_hits, f"deployed path imports torch outside the spec-mandated quorum tail: {torch_hits}"
    assert not hits, f"deployed path introduced training/CTDE-model/v5 patterns: {hits}"
