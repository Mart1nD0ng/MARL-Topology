import importlib.util
import json
from pathlib import Path

from marl_topology.training.critic_repair_trainer import (
    Stage27CriticRepairTrainConfig,
    run_stage27_critic_repair,
)

from stage27_synthetic import make_stage27_synthetic_dataset


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "train" / "stage27_critic_baseline_repair.py"
REPORT_PATH = (
    ROOT
    / "result_save"
    / "stage27_critic_baseline_repair"
    / "stage27_critic_repair_dataset_and_train_config"
    / "stage27_critic_repair_report.json"
)


def test_stage27_manifest_blocks_actor_updates_checkpoints_and_scale_up() -> None:
    script = _load_stage27_script()
    manifest = script.build_stage27_manifest()

    assert manifest["stage_id"] == "stage_27_critic_baseline_repair"
    assert manifest["artifact_write_allowed"] == "manifest_validated_reports_only"
    assert manifest["checkpoint_creation_allowed"] is False
    assert manifest["training_scale_up_allowed"] is False
    assert manifest["actor_update_allowed"] is False
    assert manifest["policy_gradient_update_allowed"] is False


def test_stage27_report_records_no_actor_update_or_forbidden_action() -> None:
    report = _load_report_or_synthetic()

    assert report["forbidden_action_flags"] == {
        "actor_update_performed": False,
        "policy_gradient_update_performed": False,
        "scale_up_training_run": False,
        "reward_weights_changed": False,
        "sampler_switched": False,
        "checkpoint_created": False,
        "legacy_v5_modified": False,
    }
    assert report["dataset_health"]["actor_update_performed"] is False
    assert report["dataset_health"]["policy_gradient_update_performed"] is False
    assert report["mappo_readiness"]["actor_update_performed"] is False
    assert report["mappo_readiness"]["policy_gradient_update_performed"] is False


def test_stage27_sources_do_not_call_actor_training_or_checkpoint_io() -> None:
    paths = [
        ROOT / "src" / "marl_topology" / "training" / "critic_dataset.py",
        ROOT / "src" / "marl_topology" / "training" / "critic_repair_trainer.py",
        SCRIPT_PATH,
    ]
    forbidden_text = (
        "run_stage25_small_scale_mappo_training_pilot",
        "run_stage25_small_scale_formal_mappo_pilot",
        "clipped_policy_value_loss",
        "torch.save",
        "torch.load",
        "checkpoint_path",
        "training_scale_up_allowed\": true",
        "actor_update_allowed\": true",
    )

    hits: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT).as_posix()
        hits.extend(f"{relative}:{pattern}" for pattern in forbidden_text if pattern in text)

    assert hits == []


def _load_stage27_script():
    spec = importlib.util.spec_from_file_location("stage27_script", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_report_or_synthetic() -> dict[str, object]:
    if REPORT_PATH.exists():
        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    return run_stage27_critic_repair(
        dataset=make_stage27_synthetic_dataset(),
        config=Stage27CriticRepairTrainConfig(
            epochs=2,
            min_eval_explained_variance=-10.0,
            min_eval_value_return_correlation=-10.0,
            min_bias_reduction_fraction=-10.0,
            min_advantage_variance_reduction=-10.0,
        ),
    )
