"""Deployment-layer model/training/migration purity gate (was the Stage 8 policy-architecture
+ assembler contract; doc lineage retired).

Complements the Stage 9.0 / Stage 8.0 deployment-purity scans: the deployed paths (everything
under ``src/marl_topology`` EXCEPT ``models/`` and ``training/``) carry no tensorflow, no
COMA/PPO/MAPPO model classes, no training loop, no checkpoint persistence, and no v5 migration
lineage. Those live legally under the CTDE ``models/`` and ``training/`` subtrees (Engineering-
Plan Phases 7-13). The stale architecture-decision / assembler-contract doc, PROJECT_STATE, and
harness-task assertions were retired on 2026-06-23 with the multi-stage process lineage; this
real source-purity scan survives.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_CTDE_TRAINING_SUBTREES = {
    ("src", "marl_topology", "models"),
    ("src", "marl_topology", "training"),
}


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
        if relative.parts[:3] in _CTDE_TRAINING_SUBTREES:
            # CTDE training subtrees: MAPPO/COMA/PPO model classes + training loop +
            # checkpoint persistence are legal here (Engineering-Plan Phases 7-13).
            continue
        text = path.read_text(encoding="utf-8")
        # The Technical-Spec mandates a differentiable Torch quorum tail at
        # protocol/torch_quorum_tail.py (Spec S4.4-4.5); it is a standalone submodule, so the
        # base protocol package stays Torch-free. Every other deployed path imports no torch.
        torch_allowed = relative.as_posix() == "src/marl_topology/protocol/torch_quorum_tail.py"
        if ("import torch" in text or "from torch" in text) and not torch_allowed:
            torch_hits.append(relative.as_posix())
        for term in banned_terms:
            if term in text:
                hits.append(f"{relative}:{term}")
    assert not torch_hits, f"deployed path imports torch outside the spec-mandated quorum tail: {torch_hits}"
    assert not hits, f"deployed path introduced model/training/migration patterns: {hits}"
