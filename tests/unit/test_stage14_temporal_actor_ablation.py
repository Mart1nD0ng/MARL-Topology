from pathlib import Path

import torch

from scripts.train.stage14_temporal_actor_ablation import build_report

from marl_topology.models import (
    LOCAL_GNN_EDGE_SCORER_MODEL_ID,
    LocalGRUEdgeScorer,
    LocalLSTMEdgeScorer,
)
from marl_topology.training.sequence_batching import (
    assert_no_future_outcome_leakage,
    build_actor_safe_temporal_fixture_from_evidence,
)


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = (
    ROOT
    / "result_save"
    / "evidence_dataset_only"
    / "stage7_completion_learning_evidence_dataset_v1"
    / "learning_evidence.json"
)


def test_stage14_sequence_batching_preserves_time_order_and_no_future_leakage() -> None:
    batch = build_actor_safe_temporal_fixture_from_evidence(EVIDENCE_PATH, time_steps=4)

    assert batch.time_steps == (0, 1, 2, 3)
    assert batch.mask.shape == batch.targets.shape
    assert_no_future_outcome_leakage(batch)


def test_stage14_gru_and_lstm_output_edge_scores_and_reset_hidden() -> None:
    batch = build_actor_safe_temporal_fixture_from_evidence(EVIDENCE_PATH, time_steps=3)
    gru = LocalGRUEdgeScorer()
    lstm = LocalLSTMEdgeScorer()

    gru_hidden = gru.reset_hidden(batch.sequence_count)
    gru_logits, _ = gru(batch.features, gru_hidden)
    lstm_hidden = lstm.reset_hidden(batch.sequence_count)
    lstm_logits, _ = lstm(batch.features, lstm_hidden)

    assert gru_logits.shape == batch.targets.shape
    assert lstm_logits.shape == batch.targets.shape
    assert gru.boundary_report()["outputs_edge_scores_only"] is True
    assert lstm.boundary_report()["future_outcomes_used"] is False


def test_stage14_variable_sequence_mask_supported() -> None:
    batch = build_actor_safe_temporal_fixture_from_evidence(EVIDENCE_PATH, time_steps=4)
    mask = batch.mask.clone()
    mask[0, -1] = False

    assert mask.sum().item() == batch.mask.sum().item() - 1
    assert torch.isfinite(batch.features[mask]).all().item()


def test_stage14_report_tests_lstm_after_gru_and_keeps_stage15_conservative() -> None:
    report = build_report()

    assert report["gru_sanity_passed"] is True
    assert report["lstm_tested_after_gru"] is True
    assert report["future_outcome_leakage_detected"] is False
    assert report["selected_stage15_actor"] == LOCAL_GNN_EDGE_SCORER_MODEL_ID


def test_stage14_sources_avoid_transformer_policy_gradient_and_checkpoints() -> None:
    paths = [
        ROOT / "src" / "marl_topology" / "models" / "local_gru_edge_scorer.py",
        ROOT / "src" / "marl_topology" / "models" / "local_lstm_edge_scorer.py",
        ROOT / "src" / "marl_topology" / "training" / "sequence_batching.py",
        ROOT / "scripts" / "train" / "stage14_temporal_actor_ablation.py",
    ]
    banned_terms = [
        "torch.save",
        "torch.load",
        "checkpoint_path",
        "class PPO",
        "class MAPPO",
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
