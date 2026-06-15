from pathlib import Path

from scripts.train.stage15_controlled_ppo_pilot import (
    build_report,
    build_stage15_manifest,
)

from marl_topology.training.run_manifest_validator import validate_run_manifest_dry_run
from marl_topology.models import LOCAL_GNN_EDGE_SCORER_MODEL_ID


ROOT = Path(__file__).resolve().parents[2]


def test_stage15_policy_pilot_report_has_required_metrics_and_semantics() -> None:
    report = build_report()
    result = report["pilot_result"]

    assert report["logprob_semantics"]["projected_topology_logprob_exact"] is False
    assert "consensus_success_probability" in result["before"]
    assert "violation_rate" in result["after"]
    assert "projection_rejection_rate" in result
    assert "actor_score_distribution" in result
    assert report["coma_introduced"] is False
    assert report["transformer_introduced"] is False
    assert report["checkpoint_written"] is False


def test_stage15_manifest_records_actor_reward_and_artifact_policy() -> None:
    manifest = build_stage15_manifest("stage15_test_manifest")
    validation = validate_run_manifest_dry_run(manifest, project_root=ROOT)

    assert validation.is_valid
    assert manifest["model_id"] == LOCAL_GNN_EDGE_SCORER_MODEL_ID
    assert manifest["reward_id"] == "stage15_pilot_equal_component_weights_not_calibrated"
    assert manifest["artifact_policy_id"] == "run_manifest_artifact_contract_stage5_9"


def test_stage15_sources_avoid_coma_transformer_checkpoints_and_v5() -> None:
    paths = [
        ROOT / "src" / "marl_topology" / "training" / "ppo_pilot.py",
        ROOT / "scripts" / "train" / "stage15_controlled_ppo_pilot.py",
    ]
    banned_terms = [
        "torch.save",
        "torch.load",
        "checkpoint_path",
        "class COMA",
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
