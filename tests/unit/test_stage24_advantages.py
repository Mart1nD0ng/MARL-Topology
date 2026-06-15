import torch

from marl_topology.training.mappo.advantages import AdvantageConfig, compute_gae_returns


def test_stage24_gae_returns_are_finite_and_normalized() -> None:
    rewards = torch.tensor([1.0, 0.5, -0.25, 0.25, 1.5, -0.5])
    values = torch.tensor([0.2, 0.1, 0.0, 0.3, 0.2, -0.1])
    masks = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 0.0])

    result = compute_gae_returns(
        rewards,
        values,
        masks,
        num_scenarios=2,
        rollout_steps=3,
        config=AdvantageConfig(normalize_advantages=True),
    )

    assert result.value_baseline_used is True
    assert result.normalized is True
    assert result.advantages.shape == rewards.shape
    assert result.returns.shape == rewards.shape
    assert torch.isfinite(result.advantages).all().item()
    assert torch.isfinite(result.returns).all().item()
    assert abs(float(result.advantages.mean().item())) < 1e-6


def test_stage24_gae_masks_stop_bootstrap_after_done() -> None:
    rewards = torch.tensor([1.0, 10.0])
    values = torch.tensor([0.5, 100.0])
    masks = torch.tensor([0.0, 0.0])

    result = compute_gae_returns(
        rewards,
        values,
        masks,
        num_scenarios=1,
        rollout_steps=2,
        config=AdvantageConfig(normalize_advantages=False),
    )

    assert torch.allclose(result.raw_advantages[0], rewards[0] - values[0])
    assert torch.allclose(result.returns[0], rewards[0])
