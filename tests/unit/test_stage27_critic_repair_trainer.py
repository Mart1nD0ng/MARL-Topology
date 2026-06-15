import pytest

from marl_topology.models import (
    CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID,
    ENRICHED_CENTRALIZED_MLP_CRITIC_ID,
)
from marl_topology.training.critic_repair_trainer import (
    Stage27CriticRepairTrainConfig,
    denormalize_values_for_gae,
    run_stage27_critic_repair,
)

from stage27_synthetic import make_stage27_synthetic_dataset


def test_stage27_critic_repair_trainer_is_critic_only_and_selects_one_candidate() -> None:
    dataset = make_stage27_synthetic_dataset()
    report = run_stage27_critic_repair(
        dataset=dataset,
        config=Stage27CriticRepairTrainConfig(
            epochs=4,
            min_eval_explained_variance=-10.0,
            min_eval_value_return_correlation=-10.0,
            min_bias_reduction_fraction=-10.0,
            min_advantage_variance_reduction=-10.0,
        ),
    )

    assert report["pass_gate"] is True
    assert report["exactly_one_active_value_critic_selected"] is True
    assert report["selected_critic_id"] in {
        ENRICHED_CENTRALIZED_MLP_CRITIC_ID,
        CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID,
    }
    assert report["return_normalizer_report"]["fitted_on_train_only"] is True
    assert report["mappo_readiness"]["gae_uses_raw_denormalized_values"] is True
    assert report["forbidden_action_flags"] == {
        "actor_update_performed": False,
        "policy_gradient_update_performed": False,
        "scale_up_training_run": False,
        "reward_weights_changed": False,
        "sampler_switched": False,
        "checkpoint_created": False,
        "legacy_v5_modified": False,
    }
    assert all(candidate["actor_update_performed"] is False for candidate in report["candidate_reports"])
    assert all(candidate["implemented"] is True for candidate in report["candidate_reports"])


def test_denormalized_future_gae_path_is_available_from_trainer_report() -> None:
    dataset = make_stage27_synthetic_dataset()
    report = run_stage27_critic_repair(
        dataset=dataset,
        config=Stage27CriticRepairTrainConfig(
            epochs=2,
            min_eval_explained_variance=-10.0,
            min_eval_value_return_correlation=-10.0,
            min_bias_reduction_fraction=-10.0,
            min_advantage_variance_reduction=-10.0,
        ),
    )
    normalizer_state = report["return_normalizer"]
    from marl_topology.training.return_normalization import ReturnNormalizer
    import torch

    normalizer = ReturnNormalizer.from_state_dict(normalizer_state)
    raw = denormalize_values_for_gae(torch.tensor([0.0]), normalizer)

    assert raw.item() == pytest.approx(normalizer.mean)
