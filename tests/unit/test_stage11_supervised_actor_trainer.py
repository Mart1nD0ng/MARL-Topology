import json
from pathlib import Path

from scripts.train.stage11_supervised_mlp_actor import build_stage11_manifest

from marl_topology.training.run_manifest_validator import validate_run_manifest_dry_run
from marl_topology.training.supervised_actor_trainer import (
    SupervisedActorTrainingConfig,
    run_supervised_actor_training,
)


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = (
    ROOT
    / "result_save"
    / "evidence_dataset_only"
    / "stage7_completion_learning_evidence_dataset_v1"
    / "learning_evidence.json"
)


def _rows() -> tuple[dict[str, object], ...]:
    data = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    return tuple(dict(row) for row in data["rows"])


def test_stage11_supervised_actor_training_updates_actor_only() -> None:
    config = SupervisedActorTrainingConfig(epochs=8, tiny_batch_epochs=80)
    _model, result = run_supervised_actor_training(_rows()[:8], config=config)

    assert result.actor_parameter_update_performed is True
    assert result.update_steps == 8
    assert result.critic_training_performed is False
    assert result.rl_training_performed is False
    assert result.checkpoint_written is False
    assert result.leakage_check_passed is True
    assert result.validation_loss >= 0.0
    assert "consensus_success_probability" in result.assembler_diagnostics


def test_stage11_tiny_batch_overfit_passes_on_deterministic_row() -> None:
    config = SupervisedActorTrainingConfig(
        epochs=2,
        tiny_batch_epochs=260,
        tiny_batch_learning_rate=0.04,
        tiny_overfit_loss_threshold=0.12,
    )
    _model, result = run_supervised_actor_training(_rows()[:6], config=config)

    assert result.tiny_batch_final_loss < result.tiny_batch_initial_loss
    assert result.tiny_batch_overfit_passed is True


def test_stage11_manifest_records_required_training_ids() -> None:
    manifest = build_stage11_manifest("stage11_test_manifest")
    validation = validate_run_manifest_dry_run(manifest, project_root=ROOT)

    assert validation.is_valid
    assert manifest["model_id"] == "local_mlp_edge_scorer_stage11_supervised"
    assert manifest["data_ids"] == ["stage7_completion_learning_evidence_dataset_v1"]
    assert manifest["reward_id"] == "not_used_supervised_actor_warm_start"
    assert str(manifest["artifact_root"]).startswith("result_save")


def test_stage11_sources_avoid_forbidden_architectures_and_checkpoints() -> None:
    paths = [
        ROOT / "src" / "marl_topology" / "training" / "supervised_actor_trainer.py",
        ROOT / "scripts" / "train" / "stage11_supervised_mlp_actor.py",
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
