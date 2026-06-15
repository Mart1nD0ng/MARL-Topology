import json
from pathlib import Path

import pytest

from marl_topology.models import (
    CENTRALIZED_MLP_CRITIC_BASELINE_ID,
    CRITIC_HEAD_NAMES,
    LOCAL_MLP_EDGE_SCORER_MODEL_ID,
    CentralizedMLPCriticBaseline,
    CentralizedMLPCriticConfig,
    LocalMLPEdgeScorer,
    LocalMLPEdgeScorerConfig,
    TensorizerViolation,
    build_model_registry,
    tensorize_actor_policy_inputs,
    tensorize_actor_policy_rows,
    tensorize_critic_evidence_rows,
)
from marl_topology.policies import (
    ACTOR_EDGE_SCORE_OUTPUT_SCHEMA_ID,
    ACTOR_POLICY_OUTPUT_SCHEMA_ID,
    ActorPolicyInput,
    ConflictAwareGreedyAssembler,
)
from marl_topology.policies.topology_assembler import CandidateEdgeConstraint


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = (
    ROOT
    / "result_save"
    / "evidence_dataset_only"
    / "stage7_completion_learning_evidence_dataset_v1"
    / "learning_evidence.json"
)


def _row() -> dict[str, object]:
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))["rows"][0]


def _policy_inputs() -> tuple[ActorPolicyInput, ...]:
    return tuple(
        ActorPolicyInput.from_actor_safe_row(row)
        for row in _row()["actor_safe_rows"]  # type: ignore[index]
    )


def test_stage9_actor_output_schema_is_edge_score_not_activate() -> None:
    assert ACTOR_POLICY_OUTPUT_SCHEMA_ID == ACTOR_EDGE_SCORE_OUTPUT_SCHEMA_ID
    batch = tensorize_actor_policy_inputs(_policy_inputs())
    actor = LocalMLPEdgeScorer(LocalMLPEdgeScorerConfig(freeze_parameters=True))
    logits = actor.score_tensor_batch(batch)
    scores = batch.to_edge_score_batch(
        logits,
        batch_id="stage9_test_scores",
        source="stage9_test",
    )
    payload = scores.to_payload()

    assert payload["contains_final_topology"] is False
    assert "activate" not in payload["field_names"]
    assert all(record.score_source == "stage9_test" for record in scores.edge_scores)


def test_stage9_actor_tensorizer_rejects_forbidden_fields() -> None:
    row = dict(_row()["actor_safe_rows"][0])  # type: ignore[index]
    row["consensus_success_probability"] = 1.0

    with pytest.raises(TensorizerViolation):
        tensorize_actor_policy_rows((row,))


def test_stage9_actor_scores_are_consumed_by_conflict_aware_assembler() -> None:
    batch = tensorize_actor_policy_inputs(_policy_inputs())
    actor = LocalMLPEdgeScorer(LocalMLPEdgeScorerConfig(freeze_parameters=True))
    scores = batch.to_edge_score_batch(
        actor.score_tensor_batch(batch),
        batch_id="stage9_assembler_test",
        source="stage9_test",
    )
    constraints = tuple(
        CandidateEdgeConstraint(
            edge_id=record.edge_id,
            tx_id=record.agent_id,
            rx_id=record.neighbor_id,
            edge_type="actor_local_candidate",
            role_allowed=True,
            channel_slot=None,
            conflict_group=None,
        )
        for record in scores.edge_scores
    )

    assembled = ConflictAwareGreedyAssembler().assemble(scores.edge_scores, constraints)

    assert assembled.pre_projection_edge_count == len(scores.edge_scores)
    assert assembled.diagnostics["recommended_for_deployment"] is True
    assert assembled.diagnostics["oracle_used"] is False
    assert assembled.diagnostics["objective_used"] is False


def test_stage9_centralized_critic_heads_exist_and_are_training_only() -> None:
    critic_batch = tensorize_critic_evidence_rows((_row(),))
    critic = CentralizedMLPCriticBaseline(
        CentralizedMLPCriticConfig(edge_output_dim=critic_batch.edge_count)
    )
    output = critic.predict_tensor_batch(critic_batch)

    assert output.training_only is True
    assert output.head_names == CRITIC_HEAD_NAMES
    output.assert_shapes(batch_size=critic_batch.batch_size, edge_count=critic_batch.edge_count)


def test_stage9_model_registry_declares_actor_and_critic_boundaries() -> None:
    registry = build_model_registry()

    actor = registry[LOCAL_MLP_EDGE_SCORER_MODEL_ID]
    critic = registry[CENTRALIZED_MLP_CRITIC_BASELINE_ID]
    assert actor.training_only is False
    assert actor.output_schema_id == "actor_policy_local_edge_score_output_v1"
    assert "selected_topology" in actor.forbidden_exports
    assert critic.training_only is True
    assert critic.role == "training_only_centralized_critic"
