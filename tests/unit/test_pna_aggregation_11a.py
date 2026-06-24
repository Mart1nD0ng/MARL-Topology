"""Phase 11a (Spec §7.12): Principal Neighbourhood Aggregation primitives (Corso 2020).

Pins the 4 aggregators (mean/max/min/std) against independent ground truth, the degree scalers
S(d,alpha)=(log(1+d)/delta)^alpha (amplify/identity/attenuate), the full (aggregator x scaler) PNA
readout shape + values, permutation invariance, the isolated-node (k=0) and single-neighbour (k=1)
edge cases, and device/dtype preservation.
"""

from __future__ import annotations

import math

import pytest
import torch

from marl_topology.models.pna_aggregation import (
    PNA_SCALER_ALPHAS,
    pna_aggregators,
    pna_combine,
    pna_degree_scalers,
    training_degree_delta,
)


def test_aggregators_match_independent_ground_truth() -> None:
    m = torch.tensor([[1.0, 4.0], [3.0, 0.0], [2.0, 8.0]])
    agg = pna_aggregators(m)                       # [4, 2] = mean, max, min, std
    assert torch.allclose(agg[0], torch.tensor([2.0, 4.0]))                 # mean
    assert torch.allclose(agg[1], torch.tensor([3.0, 8.0]))                 # max
    assert torch.allclose(agg[2], torch.tensor([1.0, 0.0]))                 # min
    # population std (unbiased=False)
    col0 = torch.tensor([1.0, 3.0, 2.0]); col1 = torch.tensor([4.0, 0.0, 8.0])
    assert torch.allclose(agg[3], torch.stack([col0.std(unbiased=False), col1.std(unbiased=False)]))


def test_aggregators_permutation_invariant() -> None:
    m = torch.randn(7, 5)
    perm = m[torch.randperm(7)]
    assert torch.allclose(pna_aggregators(m), pna_aggregators(perm), atol=1e-6)


def test_aggregators_edge_cases() -> None:
    assert torch.equal(pna_aggregators(torch.zeros(0, 3)), torch.zeros(4, 3))   # isolated node
    single = pna_aggregators(torch.tensor([[2.0, -1.0]]))                       # k=1
    assert torch.allclose(single[0], torch.tensor([2.0, -1.0]))                 # mean==max==min==value
    assert torch.allclose(single[1], torch.tensor([2.0, -1.0]))
    assert torch.allclose(single[2], torch.tensor([2.0, -1.0]))
    assert torch.allclose(single[3], torch.zeros(2))                            # std of one element == 0 (no NaN)


def test_degree_scalers_amplify_identity_attenuate() -> None:
    delta = 1.5
    deg = torch.tensor([0.0, 3.0, 10.0])
    s = pna_degree_scalers(deg, delta, alphas=(1.0, 0.0, -1.0))   # [3, 3]
    # identity column is exactly 1 for every node
    assert torch.allclose(s[:, 1], torch.ones(3))
    # amplification column == log(1+d)/delta
    assert torch.allclose(s[:, 0], torch.tensor([math.log1p(d) / delta for d in (0.0, 3.0, 10.0)]))
    # attenuation == reciprocal of amplification (where defined)
    assert s[1, 2] == pytest.approx(1.0 / (math.log1p(3.0) / delta))
    # d=0 -> log1p=0 -> amplify 0, attenuate inf; identity still 1 (we only rely on identity at d=0)
    assert s[0, 0] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        pna_degree_scalers(deg, delta=0.0)


def test_pna_combine_shape_and_outer_product() -> None:
    n, n_agg, f = 2, 4, 5
    aggregated = torch.randn(n, n_agg, f)
    degree = torch.tensor([2.0, 6.0])
    delta = training_degree_delta([1, 2, 3, 6])
    out = pna_combine(aggregated, degree, delta, alphas=PNA_SCALER_ALPHAS)
    assert out.shape == (n, n_agg * len(PNA_SCALER_ALPHAS) * f)               # 4*3*5 = 60
    # spot-check the outer product: out[node, agg, scaler, :] == aggregated[node,agg] * scaler
    scalers = pna_degree_scalers(degree, delta)
    recon = out.reshape(n, n_agg, len(PNA_SCALER_ALPHAS), f)
    assert torch.allclose(recon[0, 1, 2], aggregated[0, 1] * scalers[0, 2], atol=1e-6)


def test_training_degree_delta() -> None:
    assert training_degree_delta([0, 0]) == pytest.approx(0.0 if False else math.log1p(0))  # 0
    assert training_degree_delta([1, 3]) == pytest.approx((math.log1p(1) + math.log1p(3)) / 2)
    assert training_degree_delta([]) == 1.0


def test_device_dtype_preserving() -> None:
    m = torch.randn(4, 3, dtype=torch.float64)
    assert pna_aggregators(m).dtype == torch.float64
    if torch.cuda.is_available():
        agg = pna_aggregators(torch.randn(4, 3, device="cuda"))
        assert agg.device.type == "cuda"
        sc = pna_degree_scalers(torch.tensor([2.0, 3.0], device="cuda"), 1.2)
        assert sc.device.type == "cuda"
