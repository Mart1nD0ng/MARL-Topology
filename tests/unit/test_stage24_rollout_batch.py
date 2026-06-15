import torch

from marl_topology.training.mappo.rollout import RolloutTransition, build_rollout_batch


def _transition(index: int, *, done: bool) -> RolloutTransition:
    return RolloutTransition(
        actor_safe_observation={
            "view_role": "actor_safe_decentralized_policy_input",
            "field_names": ["agent_id", "local_neighbor_observations"],
            "forbidden_actor_fields_detected": [],
        },
        centralized_critic_input={
            "critic_view": {
                "view_role": "critic_centralized_training_only",
                "selected_edge_ids": ("a--b",),
            }
        },
        actor_logits=(0.1, 0.2),
        sampler_id="physical_plackett_luce_top_k_sampler",
        proposal_action={"policy_action": "proposal", "proposed_physical_edges": ("a--b",)},
        proposal_logprob=torch.tensor(-1.0),
        proposal_entropy=torch.tensor(0.5),
        pre_projection_proposals=("a--b",),
        post_projection_selected_physical_edges=("a--b",),
        projection_diagnostics={
            "top_proposal_rejection_rate": 0.0,
            "above_threshold_rejection_rate": 0.0,
            "rejection_by_reason": {},
            "selected_edge_count": 1,
            "candidate_physical_edge_count": 2,
        },
        consensus_success_probability=0.9,
        latency=0.001,
        energy=0.002,
        reward_surrogate=1.0,
        value_prediction=torch.tensor(0.25),
        done=done,
        mask=0.0 if done else 1.0,
        scenario_id="fixture",
        time_step=index,
        seed=24 + index,
        row_index=0,
        step_index=index,
        raw_sample_data={"ordered_indices": (0,)},
    )


def test_stage24_rollout_batch_shapes_masks_and_actor_boundary() -> None:
    batch = build_rollout_batch(
        (_transition(0, done=False), _transition(1, done=True)),
        num_scenarios=1,
        rollout_steps=2,
    )

    assert batch.total_transitions == 2
    assert batch.old_logprobs.shape == (2,)
    assert batch.rewards.shape == (2,)
    assert batch.values.shape == (2,)
    assert batch.masks.tolist() == [1.0, 0.0]
    assert batch.dones.tolist() == [0.0, 1.0]
    assert batch.observations[0]["forbidden_actor_fields_detected"] == []
    assert batch.centralized_inputs[0]["critic_view"]["view_role"] == (
        "critic_centralized_training_only"
    )
