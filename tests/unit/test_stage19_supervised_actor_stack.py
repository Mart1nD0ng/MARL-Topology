import torch

from marl_topology.data.learning_evidence_stage18 import (
    build_stage18_learning_evidence_dataset,
)
from marl_topology.training.stage19_supervised_actor_stack import (
    STAGE19_FEATURE_FIELDS,
    Stage19SupervisedActorStackConfig,
    build_stage19_actor_batch,
    build_stage19_pair_indices,
    build_stage19_real_multistep_temporal_batch,
    run_stage19_supervised_actor_stack,
)


def test_stage19_actor_batch_uses_stage18_disambiguated_actor_safe_features() -> None:
    dataset = build_stage18_learning_evidence_dataset()
    batch = build_stage19_actor_batch(dataset)

    assert len(batch.samples) == 850
    assert batch.features.shape == (850, len(STAGE19_FEATURE_FIELDS))
    assert len(STAGE19_FEATURE_FIELDS) == 31
    assert torch.all(batch.targets >= 0.0)
    assert torch.all(batch.targets <= 1.0)
    assert torch.all(batch.confidence > 0.0)

    forbidden = {
        "oracle",
        "global_topology",
        "consensus_success_probability",
        "reward_surrogate",
        "delta_consensus_success_probability",
        "delta_latency",
        "delta_energy",
        "future_outcome",
    }
    schema_text = " ".join(STAGE19_FEATURE_FIELDS)
    assert not [term for term in forbidden if term in schema_text]


def test_stage19_pairwise_ranking_pairs_are_mapped_from_stage18_targets() -> None:
    dataset = build_stage18_learning_evidence_dataset()
    batch = build_stage19_actor_batch(dataset)
    pairs = build_stage19_pair_indices(dataset, batch)

    assert len(pairs) == 412
    preferred, less_preferred, margin, confidence = pairs[0]
    assert preferred != less_preferred
    assert margin >= 0.0
    assert confidence > 0.0


def test_stage19_real_multistep_temporal_batch_uses_real_sequence_subset() -> None:
    batch = build_stage19_actor_batch(build_stage18_learning_evidence_dataset())
    temporal = build_stage19_real_multistep_temporal_batch(batch)
    summary = temporal.summary()

    assert summary["real_multistep_actor_safe_sequence_used"] is True
    assert summary["sequence_count"] == 72
    assert summary["mask_active_count"] == 216
    assert summary["time_steps"] == [0, 1, 2]
    assert summary["future_outcome_leakage_detected"] is False


def test_stage19_supervised_actor_stack_order_and_boundaries() -> None:
    report = run_stage19_supervised_actor_stack(
        config=Stage19SupervisedActorStackConfig(epochs=5, temporal_epochs=5)
    )

    assert report["rerun_order"] == ["MLP", "GNN", "GRU", "LSTM_after_GRU_sanity"]
    assert report["sample_count"] == 850
    assert report["ranking_pair_count"] == 412
    assert report["mlp"]["actor_parameter_update_performed"] is True
    assert report["gnn"]["actor_parameter_update_performed"] is True
    assert report["gru_sanity_passed"] is True
    assert report["lstm_tested_after_gru"] is True
    assert report["lstm"]["model_label"] == "LSTM"

    assert report["critic_training_performed"] is False
    assert report["policy_gradient_performed"] is False
    assert report["checkpoint_written"] is False
    assert report["artifact_written"] is False
    assert report["ppo_mappo_allowed"] is False
    assert report["coma_allowed"] is False
    assert report["transformer_allowed"] is False
    assert report["scale_up_training_allowed"] is False
    assert report["reward_weight_tuning_performed"] is False
    assert report["final_tau_selected"] is False
    assert report["v5_modified"] is False
