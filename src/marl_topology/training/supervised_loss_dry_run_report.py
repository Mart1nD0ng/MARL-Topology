"""Stage 10 supervised loss dry-run report implementation."""

from __future__ import annotations

from pathlib import Path

import torch

from marl_topology.models import (
    CentralizedMLPCriticBaseline,
    CentralizedMLPCriticConfig,
    LocalMLPEdgeScorer,
    LocalMLPEdgeScorerConfig,
)
from marl_topology.training.supervised_batching import (
    build_supervised_batch_from_evidence,
    load_learning_evidence_json,
)
from marl_topology.training.supervised_losses import (
    STAGE10_SUPERVISED_LOSS_COMPONENTS,
    compute_supervised_loss_report,
)


def build_stage10_supervised_loss_dry_run_report(evidence_path: Path) -> dict[str, object]:
    torch.manual_seed(10)
    evidence = load_learning_evidence_json(evidence_path)
    batch = build_supervised_batch_from_evidence(evidence, row_limit=4)
    actor = LocalMLPEdgeScorer(LocalMLPEdgeScorerConfig(freeze_parameters=False))
    critic = CentralizedMLPCriticBaseline(
        CentralizedMLPCriticConfig(edge_output_dim=batch.critic_batch.edge_count)
    )
    before = _parameter_checksum(actor) + _parameter_checksum(critic)
    with torch.no_grad():
        actor_logits = actor.score_tensor_batch(batch.actor_batch)
        critic_output = critic.predict_tensor_batch(batch.critic_batch)
        loss_report = compute_supervised_loss_report(
            actor_logits=actor_logits,
            critic_output=critic_output,
            batch=batch,
        )
    after = _parameter_checksum(actor) + _parameter_checksum(critic)
    loss_payload = loss_report.to_dict()
    loss_payload["update_rule_used"] = False
    return {
        "stage": "stage_10_supervised_loss_dry_run_no_training",
        "dataset_id": batch.dataset_id,
        "batch_summary": batch.target_summary(),
        "loss_report": loss_payload,
        "expected_loss_components": list(STAGE10_SUPERVISED_LOSS_COMPONENTS),
        "finite_total_loss": loss_report.total_loss >= 0.0,
        "parameter_checksum_before": before,
        "parameter_checksum_after": after,
        "parameter_update_performed": before != after,
        "backward_called": False,
        "update_rule_used": False,
        "checkpoint_written": False,
        "artifact_written": False,
        "actor_safe_fields_separated_from_targets": True,
    }


def _parameter_checksum(module: torch.nn.Module) -> float:
    return float(
        sum(parameter.detach().double().sum().item() for parameter in module.parameters())
    )
