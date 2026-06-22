"""Phase 1c: Torch-differentiable heterogeneous quorum-tail (Technical-Spec S4.4-4.5).

Pins numerical parity with the reference DP, exact small-N parity vs brute force, a
batch version, the analytic gradient (Spec S4.5: dQ/dp_i = P(exactly q-1 of the others)),
and autograd gradcheck.
"""

from __future__ import annotations

from itertools import product

import pytest

torch = pytest.importorskip("torch")

from marl_topology.protocol.quorum_tail import heterogeneous_quorum_tail
from marl_topology.protocol.torch_quorum_tail import torch_quorum_tail


def _brute_force_tail(probs: list[float], quorum: int) -> float:
    total = 0.0
    n = len(probs)
    for outcome in product((0, 1), repeat=n):
        if sum(outcome) >= quorum:
            p = 1.0
            for bit, prob in zip(outcome, probs):
                p *= prob if bit else (1.0 - prob)
            total += p
    return total


def _brute_force_exactly(probs: list[float], count: int) -> float:
    total = 0.0
    n = len(probs)
    for outcome in product((0, 1), repeat=n):
        if sum(outcome) == count:
            p = 1.0
            for bit, prob in zip(outcome, probs):
                p *= prob if bit else (1.0 - prob)
            total += p
    return total


# --- numerical parity with the reference DP across (n, quorum) ---

def test_matches_reference_dp_across_shapes() -> None:
    gen = torch.Generator().manual_seed(0)
    for n in (1, 3, 5, 8, 12):
        for quorum in range(0, n + 2):
            p = torch.rand(n, generator=gen, dtype=torch.float64)
            ref = heterogeneous_quorum_tail(p.tolist(), quorum)
            got = torch_quorum_tail(p, quorum)
            assert got.shape == ()
            assert float(got) == pytest.approx(ref, abs=1e-12)


def test_small_n_exact_parity_vs_brute_force() -> None:
    for probs in ([0.3], [0.9, 0.1], [0.5, 0.5, 0.5], [0.2, 0.7, 0.95, 0.4]):
        for quorum in range(0, len(probs) + 1):
            got = float(torch_quorum_tail(torch.tensor(probs, dtype=torch.float64), quorum))
            assert got == pytest.approx(_brute_force_tail(probs, quorum), abs=1e-12)


# --- batch version: leading dims are independent committees ---

def test_batch_matches_per_row() -> None:
    gen = torch.Generator().manual_seed(1)
    batch = torch.rand(7, 6, generator=gen, dtype=torch.float64)
    quorum = 4
    batched = torch_quorum_tail(batch, quorum)
    assert batched.shape == (7,)
    for i in range(7):
        assert float(batched[i]) == pytest.approx(
            heterogeneous_quorum_tail(batch[i].tolist(), quorum), abs=1e-12
        )
    # a 2-D leading batch as well
    multi = torch.rand(3, 4, 5, generator=gen, dtype=torch.float64)
    out = torch_quorum_tail(multi, 3)
    assert out.shape == (3, 4)


# --- edge cases ---

def test_edge_cases() -> None:
    p = torch.tensor([0.4, 0.6, 0.8], dtype=torch.float64)
    assert float(torch_quorum_tail(p, 0)) == 1.0          # quorum 0 always met
    assert float(torch_quorum_tail(p, 4)) == 0.0          # quorum > n impossible
    assert float(torch_quorum_tail(p, 3)) == pytest.approx(0.4 * 0.6 * 0.8, abs=1e-12)


# --- differentiability ---

def test_autograd_gradcheck() -> None:
    p = torch.rand(6, dtype=torch.float64, requires_grad=True)
    # clamp into (0,1) strictly so finite differences stay valid
    p_in = p.clamp(0.05, 0.95)
    assert torch.autograd.gradcheck(lambda x: torch_quorum_tail(x, 4), (p_in,), eps=1e-6, atol=1e-6)


def test_gradient_equals_leave_one_out_sensitivity() -> None:
    # Spec S4.5: dQ_q/dp_i = P(exactly q-1 successes among the OTHER n-1 nodes).
    probs = [0.2, 0.7, 0.5, 0.9, 0.35]
    quorum = 3
    p = torch.tensor(probs, dtype=torch.float64, requires_grad=True)
    out = torch_quorum_tail(p, quorum)
    out.backward()
    grad = p.grad
    for i in range(len(probs)):
        others = probs[:i] + probs[i + 1:]
        analytic = _brute_force_exactly(others, quorum - 1)
        assert float(grad[i]) == pytest.approx(analytic, abs=1e-10)


def test_backprops_into_an_upstream_parameter() -> None:
    logits = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    p = torch.sigmoid(logits)
    loss = -torch_quorum_tail(p, 3)  # maximize the quorum tail
    loss.backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))
    assert torch.any(logits.grad != 0)
