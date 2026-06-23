"""R6 (v2 Engineering-Plan §R6, Spec §7.2-7.10): the BCSP unordered subset policy.

Independent-truth tests (brute-force 2^m enumeration, the closed-form Bernoulli, MC frequency,
gradcheck) -- NOT wrapper-consistency. Pins: order-irrelevance, exact logp/entropy, the b>=m fast
path, the MAP == deployed decoder, sampling unbiasedness, autograd, and the production-scale
complexity (the m=15,b=64 blocker is LINEAR, not factorial; m=128,b=64 is polynomial).
"""

from __future__ import annotations

import itertools
import math
import time

import pytest
import torch

from marl_topology.training.budget_conditioned_subset import (
    log_partition,
    map_subset,
    normalized_entropy,
    num_subset_actions,
    sample_subset,
    subset_entropy,
    subset_logp,
)


def _brute_force(theta_list, budget):
    """Ground truth by direct enumeration of all |S| <= b subsets."""
    m = len(theta_list)
    subsets = [c for r in range(min(budget, m) + 1) for c in itertools.combinations(range(m), r)]
    weights = {s: math.exp(sum(theta_list[i] for i in s)) for s in subsets}
    z = sum(weights.values())
    probs = {s: w / z for s, w in weights.items()}
    entropy = -sum(p * math.log(p) for p in probs.values() if p > 0)
    return probs, math.log(z), entropy


def test_order_aliases_are_one_subset_action() -> None:
    theta = torch.tensor([0.5, -0.2, 0.8, 0.1], dtype=torch.float64)
    a = subset_logp(theta, [0, 2, 3], budget=3)
    b = subset_logp(theta, [3, 0, 2], budget=3)  # same SET, different element order
    c = subset_logp(theta, [2, 3, 0], budget=3)
    assert torch.allclose(a, b) and torch.allclose(a, c)


def test_k_equals_m_has_no_order_entropy() -> None:
    # b >= m and very large positive theta -> the policy concentrates on the SINGLE all-edges subset
    # -> subset entropy ~ 0. The retired ordered Plackett-Luce would give ~ log(m!) (m! orderings).
    theta = torch.tensor([20.0, 20.0, 20.0], dtype=torch.float64)
    h = float(subset_entropy(theta, budget=3))
    assert h < 1e-2
    assert h < math.log(math.factorial(3))  # decisively below the ordered-entropy inflation


def test_bcsp_matches_exhaustive_small_graph() -> None:
    torch.manual_seed(0)
    theta = torch.randn(5, dtype=torch.float64)
    probs, logz, ent = _brute_force(theta.tolist(), budget=3)
    assert float(log_partition(theta, 3)) == pytest.approx(logz, abs=1e-10)
    assert float(subset_entropy(theta, 3)) == pytest.approx(ent, abs=1e-10)
    for subset, p in probs.items():
        assert float(subset_logp(theta, list(subset), 3)) == pytest.approx(math.log(p), abs=1e-9)


def test_bcsp_b_ge_m_matches_bernoulli() -> None:
    theta = torch.tensor([0.7, -0.4, 1.2, 0.0], dtype=torch.float64)
    sigma = torch.sigmoid(theta)
    # independent Bernoulli logp(S) = sum_{e in S} log sigma + sum_{e not in S} log(1-sigma)
    subset = [0, 2]
    bern_logp = sum(math.log(float(sigma[i])) if i in subset else math.log(1 - float(sigma[i]))
                    for i in range(4))
    assert float(subset_logp(theta, subset, budget=4)) == pytest.approx(bern_logp, abs=1e-10)
    bern_entropy = float(sum(-(p := float(sigma[i])) * math.log(p) - (1 - p) * math.log(1 - p)
                             for i in range(4)))
    assert float(subset_entropy(theta, budget=4)) == pytest.approx(bern_entropy, abs=1e-10)


def test_bcsp_map_matches_local_decoder() -> None:
    from marl_topology.training.decentralized_action import deterministic_decentralized_action
    # 4-node graph; per-node MAP (top-b positive logits) combined by mutual acceptance must equal the
    # deployed deterministic decoder.
    edges = {"AB": ("A", "B"), "AC": ("A", "C"), "BC": ("B", "C"), "CD": ("C", "D")}
    edge_ids = ["AB", "AC", "BC", "CD"]
    logits = torch.tensor([0.9, -0.3, 0.5, 0.2], dtype=torch.float64)
    budgets = {"A": 2, "B": 2, "C": 1, "D": 2}
    # per-node MAP over incident edges
    incident = {n: [i for i, e in enumerate(edge_ids) if n in edges[e]] for n in "ABCD"}
    accept = {}
    for n, idxs in incident.items():
        local_theta = logits[idxs]
        chosen_local = map_subset(local_theta, budgets[n])
        accept[n] = {idxs[c] for c in chosen_local}
    mutual = tuple(sorted(i for i, e in enumerate(edge_ids)
                          if i in accept[edges[e][0]] and i in accept[edges[e][1]]))
    deployed = deterministic_decentralized_action(logits, edge_ids, edges=edges, budgets=budgets)
    assert mutual == tuple(sorted(deployed))


def test_bcsp_sampling_frequency_matches_probability() -> None:
    theta = torch.tensor([0.6, -0.5, 1.0], dtype=torch.float64)
    probs, _logz, _ent = _brute_force(theta.tolist(), budget=2)
    gen = torch.Generator().manual_seed(12345)
    counts: dict[tuple, int] = {}
    n = 40000
    for _ in range(n):
        s = tuple(sample_subset(theta, budget=2, generator=gen))
        counts[s] = counts.get(s, 0) + 1
    for subset, p in probs.items():
        emp = counts.get(subset, 0) / n
        assert emp == pytest.approx(p, abs=0.02), (subset, emp, p)


def test_bcsp_gradcheck() -> None:
    theta = torch.randn(5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: subset_logp(t, [0, 2, 4], budget=3), (theta,))
    theta2 = torch.randn(5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: subset_entropy(t, budget=3), (theta2,))


def test_bcsp_m15_b64_runtime() -> None:
    # THE blocker: degree m=15, radio budget b=64 -> b >= m -> the O(m) Bernoulli fast path, NOT 15!.
    theta = torch.randn(15, dtype=torch.float64)
    t0 = time.perf_counter()
    for _ in range(50):
        log_partition(theta, 64)
        subset_entropy(theta, 64)
        sample_subset(theta, 64)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0  # 50 full passes in << 1s (factorial would never finish)
    assert num_subset_actions(15, 64) == 2 ** 15  # b>=m -> every subset is legal


def test_bcsp_m128_b64_polynomial_scaling() -> None:
    # b < m -> the O(m b) DP; m=128,b=64 is ~8k cells, tractable (a factorial enumeration is not).
    theta = torch.randn(128, dtype=torch.float64)
    t0 = time.perf_counter()
    logz = log_partition(theta, 64)
    h = subset_entropy(theta, 64)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0
    assert math.isfinite(float(logz)) and math.isfinite(float(h))


def test_no_permutations_in_production_action_path() -> None:
    from pathlib import Path
    src = Path(__file__).resolve().parents[1].parent / "src" / "marl_topology" / "training" / "budget_conditioned_subset.py"
    text = src.read_text(encoding="utf-8")
    assert "itertools.permutations" not in text and "permutations(" not in text


def test_bcsp_budget_zero_is_the_empty_subset_and_differentiable() -> None:
    # b=0 (a node with no radio budget) -> only the empty subset is legal -> H=0, mu=0, logp(empty)=0.
    # Regression for the adversarial-verify finding (the b<m DP gave a graph-disconnected logZ that
    # crashed autograd.grad in subset_entropy / inclusion_marginals / normalized_entropy).
    from marl_topology.training.budget_conditioned_subset import inclusion_marginals
    theta = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)
    assert float(log_partition(theta, 0)) == pytest.approx(0.0)
    assert float(subset_entropy(theta, 0)) == pytest.approx(0.0)
    assert float(normalized_entropy(theta, 0)) == pytest.approx(0.0)
    assert torch.allclose(inclusion_marginals(theta, 0), torch.zeros_like(theta))
    assert float(subset_logp(theta, [], 0)) == pytest.approx(0.0)
    # the entropy bonus must stay differentiable at b=0 (zero gradient, no crash)
    t = theta.clone().requires_grad_(True)
    subset_entropy(t, 0).backward()
    assert t.grad is not None and torch.allclose(t.grad, torch.zeros_like(t))


def test_cuda_device_path_if_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    theta = torch.randn(10, dtype=torch.float64, device="cuda")
    assert log_partition(theta, 4).device.type == "cuda"
    assert subset_entropy(theta, 4).device.type == "cuda"
    assert all(0 <= i < 10 for i in sample_subset(theta, 4))
