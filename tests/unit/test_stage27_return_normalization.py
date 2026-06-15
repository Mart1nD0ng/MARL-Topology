import math

import pytest
import torch

from marl_topology.training.critic_repair_trainer import denormalize_values_for_gae
from marl_topology.training.return_normalization import (
    ReturnNormalizationViolation,
    ReturnNormalizer,
)


def test_return_normalizer_round_trips_raw_returns() -> None:
    returns = [-420.0, -350.0, -280.0, -210.0, -140.0]
    normalizer = ReturnNormalizer.fit(returns)

    normalized = normalizer.transform(returns)
    recovered = normalizer.inverse_transform(normalized)

    assert normalizer.fitted_on_train_only is True
    assert all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-7) for a, b in zip(returns, recovered))


def test_normalized_train_targets_are_centered_and_scaled() -> None:
    returns = [-500.0, -300.0, -100.0, -700.0]
    normalizer = ReturnNormalizer.fit(returns)
    report = normalizer.scale_report(returns)

    assert abs(float(report["normalized_mean"])) < 1e-8
    assert float(report["normalized_std"]) == pytest.approx(1.0)
    assert report["fitted_on_train_only"] is True
    assert report["gae_uses_raw_denormalized_values"] is True


def test_return_normalizer_rejects_eval_fitted_state() -> None:
    with pytest.raises(ReturnNormalizationViolation):
        ReturnNormalizer.fit([-1.0, -2.0], fitted_on_train_only=False)


def test_denormalized_values_for_gae_use_raw_return_scale() -> None:
    normalizer = ReturnNormalizer(mean=-350.0, std=100.0)
    normalized_values = torch.tensor([-1.0, 0.0, 1.0])

    raw_values = denormalize_values_for_gae(normalized_values, normalizer)

    assert torch.allclose(raw_values, torch.tensor([-450.0, -350.0, -250.0]))
