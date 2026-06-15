from pathlib import Path

from marl_topology.env import LocalNeighborObservation
from marl_topology.policies import (
    ACTOR_EDGE_SCORE_OUTPUT_SCHEMA_ID,
    ACTOR_POLICY_OUTPUT_SCHEMA_ID,
    LOCAL_MLP_EDGE_FEATURE_FIELDS,
    LOCAL_MLP_EDGE_SCORER_ID,
    LOCAL_MLP_FEATURE_SCHEMA_ID,
    STAGE9_0_LOCAL_MLP_STAGE_ID,
    ActorPolicyInput,
    LocalMLPEdgeScorer,
    LocalMLPEdgeScorerConfig,
    LocalMLPEdgeScorerViolation,
    build_stage9_0_local_mlp_report,
    encode_actor_local_edges,
    validate_actor_policy_output_fields,
)


ROOT = Path(__file__).resolve().parents[2]


def _policy_input() -> ActorPolicyInput:
    return ActorPolicyInput(
        agent_id="veh_0",
        agent_kind="vehicle",
        time_step=3,
        local_position_m=(1.0, 2.0, 1.5),
        local_neighbor_observations=(
            LocalNeighborObservation(
                neighbor_id="veh_1",
                neighbor_kind="vehicle",
                edge_id="veh_0--veh_1",
                distance_3d_m=12.0,
                link_success_probability=0.91,
                estimated_link_latency_s=0.002,
                estimated_link_energy_j=0.015,
            ),
            LocalNeighborObservation(
                neighbor_id="rsu_0",
                neighbor_kind="rsu",
                edge_id="rsu_0--veh_0",
                distance_3d_m=35.0,
                link_success_probability=0.86,
                estimated_link_latency_s=0.004,
                estimated_link_energy_j=0.021,
            ),
        ),
        local_messages=(),
        local_history={},
    )


def test_stage9_0_unifies_active_actor_output_schema_to_edge_scores() -> None:
    assert ACTOR_POLICY_OUTPUT_SCHEMA_ID == ACTOR_EDGE_SCORE_OUTPUT_SCHEMA_ID
    validate_actor_policy_output_fields(
        {
            "schema_id",
            "agent_id",
            "edge_scores",
            "policy_state_id",
            "contains_final_topology",
            "contains_oracle_label",
            "contains_consensus_metric",
        }
    )


def test_local_mlp_feature_encoder_uses_actor_safe_local_neighbor_fields() -> None:
    encoded = encode_actor_local_edges(_policy_input())

    assert len(encoded) == 2
    assert len(encoded[0].features) == len(LOCAL_MLP_EDGE_FEATURE_FIELDS)
    assert {edge.directed_edge_id for edge in encoded} == {
        "veh_0->veh_1",
        "veh_0->rsu_0",
    }
    assert LOCAL_MLP_FEATURE_SCHEMA_ID == "local_mlp_edge_feature_v1"


def test_local_mlp_edge_scorer_outputs_edge_scores_without_activate() -> None:
    scorer = LocalMLPEdgeScorer()
    output = scorer.score_actor_input(_policy_input())
    payload = output.to_payload()

    assert output.schema_id == ACTOR_EDGE_SCORE_OUTPUT_SCHEMA_ID
    assert output.agent_id == "veh_0"
    assert payload["contains_final_topology"] is False
    assert "activate" not in str(payload["edge_scores"])
    records = output.edge_scores.edge_scores
    assert len(records) == 2
    assert {record.directed_edge_id for record in records} == {
        "veh_0->veh_1",
        "veh_0->rsu_0",
    }
    assert all(record.score_source == "stage9_0_local_mlp_edge_scorer_untrained" for record in records)
    assert all(record.probability is not None for record in records)
    assert all(0.0 <= record.probability <= 1.0 for record in records if record.probability is not None)


def test_local_mlp_scaffold_is_frozen_and_training_blocked() -> None:
    scorer = LocalMLPEdgeScorer()
    boundary = scorer.training_boundary()

    assert boundary["stage"] == STAGE9_0_LOCAL_MLP_STAGE_ID
    assert boundary["scorer_id"] == LOCAL_MLP_EDGE_SCORER_ID
    assert boundary["training_execution_allowed"] is False
    assert boundary["checkpoint_io_allowed"] is False
    assert boundary["artifact_writes_allowed"] is False
    assert boundary["parameters_frozen_for_stage9"] is True
    assert all(not parameter.requires_grad for parameter in scorer.parameters())


def test_local_mlp_rejects_training_or_artifact_flags() -> None:
    for kwargs in (
        {"training_execution_allowed": True},
        {"checkpoint_io_allowed": True},
        {"artifact_writes_allowed": True},
    ):
        try:
            LocalMLPEdgeScorerConfig(**kwargs)
        except LocalMLPEdgeScorerViolation:
            pass
        else:
            raise AssertionError(f"forbidden Stage 9.0 config accepted: {kwargs}")


def test_stage9_0_report_declares_no_training_or_advanced_architecture() -> None:
    report = build_stage9_0_local_mlp_report()

    assert report["stage"] == STAGE9_0_LOCAL_MLP_STAGE_ID
    assert report["active_actor_output_schema"] == ACTOR_EDGE_SCORE_OUTPUT_SCHEMA_ID
    assert report["outputs_edge_scores_only"] is True
    assert report["activate_output_allowed"] is False
    assert report["training_execution_allowed"] is False
    assert report["checkpoint_io_allowed"] is False
    assert report["artifact_writes_allowed"] is False
    assert report["ppo_mappo_coma_implemented"] is False
    assert report["graph_recurrent_attention_modules_implemented"] is False


def test_stage9_0_source_does_not_add_training_or_checkpoint_routes() -> None:
    source = (ROOT / "src" / "marl_topology" / "policies" / "local_mlp_edge_scorer.py").read_text(
        encoding="utf-8"
    )
    banned_terms = [
        "optimizer.step",
        "backward(",
        "train_loop",
        "torch.save",
        "torch.load",
        "class PPO",
        "class MAPPO",
        "class COMA",
        "class GNN",
        "class GRU",
        "class LSTM",
        "class Transformer",
    ]
    hits = [term for term in banned_terms if term in source]
    assert not hits, f"Stage 9.0 MLP source contains forbidden terms: {hits}"
