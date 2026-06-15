import json
from pathlib import Path

from scripts.train.stage12_critic_pretraining import build_stage12_manifest

from marl_topology.training.critic_pretrainer import (
    CriticPretrainingConfig,
    pretrain_centralized_critic,
)
from marl_topology.training.run_manifest_validator import validate_run_manifest_dry_run
from marl_topology.training.supervised_batching import build_supervised_batch_from_evidence


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = (
    ROOT
    / "result_save"
    / "evidence_dataset_only"
    / "stage7_completion_learning_evidence_dataset_v1"
    / "learning_evidence.json"
)


def _evidence() -> dict[str, object]:
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def test_stage12_critic_pretraining_updates_critic_only() -> None:
    batch = build_supervised_batch_from_evidence(_evidence(), row_limit=8)
    _model, result = pretrain_centralized_critic(
        batch,
        config=CriticPretrainingConfig(epochs=20, hidden_dim=48),
    )

    assert result.critic_parameter_update_performed is True
    assert result.actor_training_performed is False
    assert result.checkpoint_written is False
    assert result.final_loss >= 0.0
    assert "add_edge_delta_ranking_spearman" in result.fidelity_report
    assert "calibration_error_by_head" in result.fidelity_report


def test_stage12_manifest_records_critic_and_data_ids() -> None:
    manifest = build_stage12_manifest("stage12_test_manifest")
    validation = validate_run_manifest_dry_run(manifest, project_root=ROOT)

    assert validation.is_valid
    assert manifest["model_id"] == "centralized_mlp_critic_stage12_pretrained"
    assert manifest["data_ids"] == ["stage7_completion_learning_evidence_dataset_v1"]
    assert manifest["reward_id"] == "not_used_critic_pretraining"


def test_stage12_sources_avoid_forbidden_policy_paths_and_checkpoints() -> None:
    paths = [
        ROOT / "src" / "marl_topology" / "training" / "critic_pretrainer.py",
        ROOT / "scripts" / "train" / "stage12_critic_pretraining.py",
    ]
    banned_terms = [
        "torch.save",
        "torch.load",
        "checkpoint_path",
        "class PPO",
        "class MAPPO",
        "class COMA",
        "class GNN",
        "class GRU",
        "class LSTM",
        "class Transformer",
        "D:\\PhD_works\\v5",
    ]
    hits = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for term in banned_terms:
            if term in text:
                hits.append(f"{path.relative_to(ROOT)}:{term}")
    assert not hits
