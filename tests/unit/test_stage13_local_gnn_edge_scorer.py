import json
from pathlib import Path

import torch

from scripts.train.stage13_supervised_gnn_actor import build_report

from marl_topology.models import LOCAL_GNN_EDGE_SCORER_MODEL_ID, LocalGNNEdgeScorer
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


def test_stage13_local_gnn_outputs_edge_scores_for_variable_groups() -> None:
    batch = build_supervised_batch_from_evidence(_evidence(), row_limit=1).actor_batch
    model = LocalGNNEdgeScorer()
    logits = model.score_tensor_batch(batch)
    edge_scores = batch.to_edge_score_batch(
        logits,
        batch_id="stage13_gnn_test",
        source="stage13_test",
    )

    assert logits.shape == (batch.edge_count,)
    assert len(set(batch.group_ids.tolist())) > 1
    assert edge_scores.to_payload()["contains_final_topology"] is False
    assert "activate" not in edge_scores.field_names


def test_stage13_gnn_group_context_is_local_to_group() -> None:
    model = LocalGNNEdgeScorer()
    features = torch.randn(5, model.config.input_dim)
    group_ids = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    logits = model.forward_with_groups(features, group_ids)

    assert logits.shape == (5,)
    report = model.boundary_report()
    assert report["global_topology_used"] is False
    assert report["outputs_edge_scores_only"] is True


def test_stage13_comparison_report_selects_stage14_starting_actor() -> None:
    report = build_report()

    assert report["global_actor_graph_used"] is False
    assert report["oracle_labels_used_as_actor_input"] is False
    assert report["critic_outputs_used_as_actor_input"] is False
    assert report["selected_stage14_starting_actor"] in {
        LOCAL_GNN_EDGE_SCORER_MODEL_ID,
        "local_mlp_edge_scorer_stage11_supervised",
    }
    assert "validation_edge_loss_delta_gnn_minus_mlp" in report["comparison"]


def test_stage13_sources_avoid_forbidden_future_architectures_and_checkpoints() -> None:
    paths = [
        ROOT / "src" / "marl_topology" / "models" / "local_gnn_edge_scorer.py",
        ROOT / "scripts" / "train" / "stage13_supervised_gnn_actor.py",
    ]
    banned_terms = [
        "torch.save",
        "torch.load",
        "checkpoint_path",
        "class PPO",
        "class MAPPO",
        "class COMA",
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
