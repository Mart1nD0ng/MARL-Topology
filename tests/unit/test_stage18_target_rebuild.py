from marl_topology.data.disambiguated_targets import (
    TARGET_ROLE_CRITIC_ONLY,
    actor_target_view_has_no_global_delta_fields,
)
from marl_topology.data.learning_evidence_stage18 import build_stage18_learning_evidence_dataset


def test_stage18_soft_utility_and_confidence_targets_are_generated() -> None:
    dataset = build_stage18_learning_evidence_dataset()
    targets = dataset.rows[0].actor_target_view["actor_soft_utility_targets"]

    assert targets
    sample = targets[0]
    assert 0.0 <= sample["actor_edge_utility_target"] <= 1.0
    assert 0.0 <= sample["actor_edge_utility_confidence"] <= 1.0
    assert sample["actor_target_source"] == "distilled_global_counterfactual"
    assert sample["actor_training_role"] == "soft_distillation_only"
    assert sample["actor_target_ambiguity_level"] in {"low", "medium", "high"}


def test_stage18_pairwise_ranking_targets_are_generated() -> None:
    dataset = build_stage18_learning_evidence_dataset()

    assert dataset.ranking_pair_count > 0
    pair = next(
        target
        for row in dataset.rows
        for target in row.actor_target_view["pairwise_ranking_targets"]
    )
    assert pair["preferred_edge_id"] != pair["less_preferred_edge_id"]
    assert pair["preference_margin"] > 0
    assert 0.0 <= pair["ranking_confidence"] <= 1.0


def test_stage18_hard_labels_are_not_allowed_for_conflicting_signatures() -> None:
    dataset = build_stage18_learning_evidence_dataset()
    hard = [
        target
        for row in dataset.rows
        for target in row.actor_target_view["hard_label_diagnostics"]
    ]

    assert hard
    assert not any(target["hard_label_allowed_for_actor_training"] for target in hard)
    assert all(target["hard_label_confidence"] == 0.0 for target in hard)


def test_stage18_global_counterfactual_targets_are_critic_only() -> None:
    dataset = build_stage18_learning_evidence_dataset()
    row = dataset.rows[0]

    assert actor_target_view_has_no_global_delta_fields(row.actor_target_view)
    critic_targets = row.critic_target_view["edge_delta_targets"]
    assert critic_targets
    assert all(target["target_role"] == TARGET_ROLE_CRITIC_ONLY for target in critic_targets)
    assert "delta_consensus_success_probability" in critic_targets[0]
    assert "delta_latency" in critic_targets[0]
    assert "delta_energy" in critic_targets[0]
    assert "delta_feasibility" in critic_targets[0]

