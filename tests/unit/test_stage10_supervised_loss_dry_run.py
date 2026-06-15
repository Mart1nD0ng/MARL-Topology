import json
from pathlib import Path

import pytest
import torch

from marl_topology.models import (
    CentralizedMLPCriticBaseline,
    CentralizedMLPCriticConfig,
    LocalMLPEdgeScorer,
)
from marl_topology.training.supervised_batching import (
    SupervisedBatchingViolation,
    build_supervised_batch_from_evidence,
    build_supervised_batch_from_evidence_rows,
)
from marl_topology.training.supervised_losses import (
    STAGE10_SUPERVISED_LOSS_COMPONENTS,
    compute_supervised_loss_report,
)


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


def test_stage10_supervised_losses_compute_finite_scalars_without_update() -> None:
    torch.manual_seed(10)
    batch = build_supervised_batch_from_evidence(_evidence(), row_limit=3)
    actor = LocalMLPEdgeScorer()
    critic = CentralizedMLPCriticBaseline(
        CentralizedMLPCriticConfig(edge_output_dim=batch.critic_batch.edge_count)
    )
    before = _checksum(actor) + _checksum(critic)
    with torch.no_grad():
        report = compute_supervised_loss_report(
            actor_logits=actor.score_tensor_batch(batch.actor_batch),
            critic_output=critic.predict_tensor_batch(batch.critic_batch),
            batch=batch,
        )
    after = _checksum(actor) + _checksum(critic)

    assert set(STAGE10_SUPERVISED_LOSS_COMPONENTS) == set(report.components)
    assert report.total_loss >= 0.0
    assert all(value >= 0.0 for value in report.components.values())
    assert before == after
    assert report.backward_called is False
    assert report.update_rule_used is False
    assert report.checkpoint_written is False


def test_stage10_batching_rejects_missing_target_fields() -> None:
    row = dict(_evidence()["rows"][0])  # type: ignore[index]
    row.pop("learning_targets")

    with pytest.raises(SupervisedBatchingViolation):
        build_supervised_batch_from_evidence_rows((row,), dataset_id="broken")


def test_stage10_actor_safe_fields_remain_separated_from_targets() -> None:
    batch = build_supervised_batch_from_evidence(_evidence(), row_limit=2)
    summary = batch.target_summary()

    assert summary["actor_safe_fields_separated_from_targets"] is True
    assert summary["targets_feed_actor_model"] is False
    assert batch.actor_edge_targets.shape[0] == batch.actor_batch.edge_count
    assert batch.critic_targets.consensus_success_probability.shape[0] == batch.critic_batch.batch_size


def test_stage10_sources_do_not_call_backward_optimizer_or_checkpoint() -> None:
    paths = [
        ROOT / "src" / "marl_topology" / "training" / "supervised_batching.py",
        ROOT / "src" / "marl_topology" / "training" / "supervised_losses.py",
        ROOT / "scripts" / "replay" / "stage10_supervised_loss_dry_run_report.py",
    ]
    banned_terms = [
        ".backward(",
        "optimizer.step",
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


def _checksum(module: torch.nn.Module) -> float:
    return float(
        sum(parameter.detach().double().sum().item() for parameter in module.parameters())
    )
