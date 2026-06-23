"""Deployment-layer reward/training purity gate (was Stage 8.0 actor-interface contract).

Complements the deployment-purity scan in the Stage 9.0 gate: the deployed paths (everything
under ``src/marl_topology`` EXCEPT the ``models/`` and ``training/`` CTDE subtrees) carry no
reward definition, no optimizer / train loop, no legacy v5 reward terms, and no COMA/MAPPO
model classes. Reward shaping + the centralized critic + MAPPO/COMA live legally under
``training/`` and ``models/`` (CTDE, Engineering-Plan Phases 7-13) and are exempt here. The
stale doc / PROJECT_STATE / harness / replay-script assertions (which asserted the now-lifted
"implementation blocked" freeze) were retired on 2026-06-23 with the multi-stage lineage.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_CTDE_TRAINING_SUBTREES = {
    ("src", "marl_topology", "models"),
    ("src", "marl_topology", "training"),
}


def test_stage8_0_source_keeps_training_model_checkpoint_v5_and_legacy_metric_out() -> None:
    banned_terms = [
        "def compute_reward",
        "class Reward",
        "reward =",
        "reward:",
        "optimizer.step",
        "train_loop",
        "D:\\PhD_works\\v5",
        "P_eff_soft",
        "P_eff_hard",
        "hard_eval",
        "soft_train",
        "class COMA",
        "class MAPPO",
    ]
    hits: list[str] = []
    for path in (ROOT / "src" / "marl_topology").rglob("*.py"):
        relative = path.relative_to(ROOT)
        if relative.parts[:3] in _CTDE_TRAINING_SUBTREES:
            # Reward shaping + centralized critic + MAPPO/COMA are legal in the CTDE
            # training subtrees (Engineering-Plan Phases 7-13). Not deployed paths.
            continue
        text = path.read_text(encoding="utf-8")
        for term in banned_terms:
            if term in text:
                hits.append(f"{relative}:{term}")
    assert not hits, f"deployed path introduced reward/training/CTDE-model/v5 terms: {hits}"
