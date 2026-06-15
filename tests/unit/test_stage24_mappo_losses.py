import torch

from marl_topology.training.mappo.losses import (
    ClippedPolicyValueLossInputs,
    clipped_policy_value_loss,
)


def test_stage24_clipped_actor_value_loss_is_finite_and_differentiable() -> None:
    new_logprobs = torch.nn.Parameter(torch.tensor([-1.0, -0.8, -1.2]))
    value_predictions = torch.nn.Parameter(torch.tensor([0.1, 0.2, 0.3]))
    result = clipped_policy_value_loss(
        ClippedPolicyValueLossInputs(
            new_logprobs=new_logprobs,
            old_logprobs=torch.tensor([-1.05, -0.75, -1.25]),
            advantages=torch.tensor([1.0, -0.5, 0.25]),
            value_predictions=value_predictions,
            returns=torch.tensor([0.4, 0.0, 0.5]),
            entropies=torch.tensor([0.7, 0.6, 0.65]),
            clip_eps=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
        )
    )

    result.total_loss.backward()

    assert torch.isfinite(result.total_loss).item()
    assert torch.isfinite(result.policy_loss).item()
    assert torch.isfinite(result.value_loss).item()
    assert 0.0 <= float(result.clip_fraction.item()) <= 1.0
    assert float(result.approx_kl.item()) >= 0.0
    assert new_logprobs.grad is not None
    assert value_predictions.grad is not None
