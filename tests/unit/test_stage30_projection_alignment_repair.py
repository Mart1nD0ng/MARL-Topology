import json
from pathlib import Path

from marl_topology.evaluation.stage30_repair_diagnostics import projection_alignment_repair


ROOT = Path(__file__).resolve().parents[2]
STAGE28_REPORT = (
    ROOT
    / "result_save"
    / "stage28_repaired_critic_mappo_rerun"
    / "stage28_repaired_critic_stage25_base_protocol"
    / "training_report.json"
)


def _stage28_report() -> dict[str, object]:
    return json.loads(STAGE28_REPORT.read_text(encoding="utf-8"))


def test_stage30_projection_diagnostic_detects_unrepaired_top_proposal_rejection() -> None:
    repair = projection_alignment_repair(_stage28_report())

    assert repair["repair_class"] == "projection_alignment"
    assert repair["top_proposal_rejection_delta"] > 0.0
    assert repair["projection_friction_decreased"] is False


def test_stage30_projection_repair_does_not_change_actor_loss_or_sampler() -> None:
    repair = projection_alignment_repair(_stage28_report())

    assert repair["active_actor_loss_changed"] is False
    assert repair["owner_activation_required"] is True
    assert "owner approval" in repair["recommended_penalty"]
