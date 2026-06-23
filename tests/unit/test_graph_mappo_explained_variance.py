"""Phase 7 headline critic metric: explained_variance = 1 - Var(r - V)/Var(r).

==1 when the critic is perfect, ==0 when it only predicts the mean (no better than the EMA
baseline), <0 when anti-correlated (a collapse alarm), and a 0.0 sentinel (NOT NaN) when
Var(r)==0 (the low-reward-variance warm-started case).
"""

from marl_topology.training.graph_mappo import approx_kl, explained_variance


def test_ev_perfect_when_value_equals_reward():
    r = [0.2, 0.5, 0.8, 0.1]
    assert abs(explained_variance(r, r) - 1.0) < 1e-12


def test_ev_zero_when_value_is_constant_mean():
    r = [0.2, 0.5, 0.8]
    mean = sum(r) / len(r)
    assert abs(explained_variance(r, [mean] * 3) - 0.0) < 1e-12


def test_ev_negative_when_anticorrelated():
    r = [0.2, 0.5, 0.8]
    mean = sum(r) / len(r)
    anti = [mean + (mean - x) for x in r]  # mirror around the mean
    assert explained_variance(r, anti) < 0.0


def test_ev_zero_sentinel_when_reward_variance_zero():
    assert explained_variance([0.5, 0.5, 0.5], [0.1, 0.9, 0.3]) == 0.0


def test_approx_kl_is_zero_for_identical_logp():
    import torch

    lp = torch.tensor([-0.4, -1.2, -0.7], dtype=torch.float64)
    assert torch.allclose(approx_kl(lp, lp), torch.zeros((), dtype=torch.float64), atol=1e-12)
    # Schulman estimator is non-negative
    lp2 = lp + torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    assert float(approx_kl(lp2, lp)) >= 0.0
