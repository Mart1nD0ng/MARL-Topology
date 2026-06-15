#!/usr/bin/env python3
"""Stage 9 forward-only model and assembler smoke report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_topology.models import (
    CentralizedMLPCriticBaseline,
    CentralizedMLPCriticConfig,
    LocalMLPEdgeScorer,
    LocalMLPEdgeScorerConfig,
    tensorize_actor_policy_inputs,
    tensorize_critic_evidence_rows,
)
from marl_topology.policies import ActorPolicyInput, ConflictAwareGreedyAssembler
from marl_topology.policies.topology_assembler import CandidateEdgeConstraint


EVIDENCE_PATH = (
    ROOT
    / "result_save"
    / "evidence_dataset_only"
    / "stage7_completion_learning_evidence_dataset_v1"
    / "learning_evidence.json"
)


def build_report() -> dict[str, object]:
    data = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    row = data["rows"][0]
    policy_inputs = tuple(
        ActorPolicyInput.from_actor_safe_row(actor_row)
        for actor_row in row["actor_safe_rows"]
    )
    actor_batch = tensorize_actor_policy_inputs(policy_inputs)
    actor = LocalMLPEdgeScorer(LocalMLPEdgeScorerConfig(freeze_parameters=True))
    logits = actor.score_tensor_batch(actor_batch)
    edge_scores = actor_batch.to_edge_score_batch(
        logits,
        batch_id="stage9_forward_smoke_actor_scores",
        source="stage9_forward_smoke",
    )
    assembler = ConflictAwareGreedyAssembler()
    assembled = assembler.assemble(
        edge_scores.edge_scores,
        _candidate_constraints(edge_scores.edge_scores),
    )

    critic_batch = tensorize_critic_evidence_rows((row,))
    critic = CentralizedMLPCriticBaseline(
        CentralizedMLPCriticConfig(edge_output_dim=critic_batch.edge_count)
    )
    critic_output = critic.predict_tensor_batch(critic_batch)
    return {
        "stage": "stage_9_forward_only_local_mlp_actor_critic_scaffold",
        "actor_forward_pass": True,
        "critic_forward_pass": True,
        "assembler_dry_run": True,
        "actor_edge_count": actor_batch.edge_count,
        "critic_batch_size": critic_batch.batch_size,
        "critic_edge_count": critic_batch.edge_count,
        "critic_heads": list(critic_output.head_names),
        "assembled_topology": {
            "assembler_id": assembled.assembler_id,
            "pre_projection_edge_count": assembled.pre_projection_edge_count,
            "post_projection_edge_count": assembled.post_projection_edge_count,
            "contains_final_topology_from_actor": False,
        },
        "training_execution_allowed": False,
        "optimizer_used": False,
        "backward_used": False,
        "checkpoint_written": False,
        "artifact_written": False,
        "v5_code_migrated": False,
    }


def _candidate_constraints(edge_scores) -> tuple[CandidateEdgeConstraint, ...]:
    constraints = []
    for score in edge_scores:
        constraints.append(
            CandidateEdgeConstraint(
                edge_id=score.edge_id,
                tx_id=score.agent_id,
                rx_id=score.neighbor_id,
                edge_type="actor_local_candidate",
                role_allowed=True,
                channel_slot=None,
                conflict_group=None,
            )
        )
    return tuple(constraints)


def main() -> int:
    print(json.dumps(build_report(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
