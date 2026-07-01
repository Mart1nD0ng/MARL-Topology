"""T2 (Temporal Recovery) — the non-saturating activation fix (task 2).

Q14/T0 defect: the residual decision logit is z = z_max*tanh(raw/z_max), which saturates (gradient -> 0 for
|raw| >> z_max), so the recurrent temporal signal cannot reach the ACTED logit once raw grows (R1 mitigated
this only indirectly, via a raw-L2 penalty that keeps raw small). T2 adds a LINEAR SKIP (a leaky-tanh):
z = z_max*tanh(raw/z_max) + residual_leak*raw. The gradient is then sech^2(raw/z_max) + residual_leak, bounded
BELOW by residual_leak > 0 -- it never vanishes, so the temporal signal always reaches the logit, regardless of
raw magnitude and without depending on the raw-L2 band-aid.

residual_leak defaults to 0.0 -> BYTE-IDENTICAL to the frozen Belief-Residual actor (zero blast radius on the
completed R1-R8 campaign). The Temporal-Recovery campaign constructs the actor with residual_leak > 0.
Fails on HEAD: BeliefResidualActor has no residual_leak parameter yet.
"""

from __future__ import annotations

import torch

from marl_topology.models.belief_residual_actor import BeliefResidualActor
from marl_topology.training.residual_saturation import recurrent_vs_memoryless_delta


def _actor(leak: float, seed: int = 0) -> BeliefResidualActor:
    torch.manual_seed(seed)
    return BeliefResidualActor(node_dim=4, edge_dim=6, hidden=8, residual_logit_scale=3.0, residual_leak=leak)


def test_leak_zero_is_byte_identical_to_tanh() -> None:
    """residual_leak=0 (the retained default) reproduces the pure tanh map exactly -> the frozen R1-R8 actor is
    untouched (back-compat)."""
    a = _actor(0.0)
    assert a.residual_leak == 0.0
    nf = torch.randn(5, 4)
    ei = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
    ef = torch.randn(3, 6)
    logits, raw, _h = a.forward(nf, ef, ei)
    s = a.residual_logit_scale
    assert torch.equal(logits, s * torch.tanh(raw / s))   # BIT-exact (leak=0 short-circuits the linear skip)
    # the short-circuit keeps leak=0 byte-identical even at the raw=+-inf boundary (no 0*inf=NaN): pure tanh -> +-s
    big = torch.tensor([float("inf"), float("-inf")])
    assert torch.equal(s * torch.tanh(big / s), torch.tensor([s, -s]))


def test_actor_applies_leaky_map() -> None:
    """residual_leak>0: the actor's forward emits z = z_max*tanh(raw/z_max) + leak*raw (load-bearing: the leaky
    skip is on the actual decision path, not just a helper)."""
    a = _actor(0.1)
    assert a.residual_leak == 0.1
    nf = torch.randn(5, 4)
    ei = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
    ef = torch.randn(3, 6)
    logits, raw, _h = a.forward(nf, ef, ei)
    s = a.residual_logit_scale
    assert torch.allclose(logits, s * torch.tanh(raw / s) + a.residual_leak * raw, atol=1e-7)


def test_leaky_map_gradient_never_vanishes() -> None:
    """The T0 vanishing-gradient defect is fixed: d z / d raw = sech^2(raw/s) + leak >= leak > 0 even for a
    saturating raw, so the temporal contribution to raw is not annihilated at the logit."""
    s, leak = 3.0, 0.1
    for r in (0.0, 5.0, 100.0, -100.0):
        raw = torch.tensor([r], requires_grad=True)
        z = s * torch.tanh(raw / s) + leak * raw
        z.backward()
        assert float(raw.grad) >= leak - 1e-6       # never vanishes (tanh alone -> ~0 at r=100)
    # contrast: the pure tanh gradient DOES vanish at large raw (the defect T2 fixes)
    raw = torch.tensor([100.0], requires_grad=True)
    (s * torch.tanh(raw / s)).backward()
    assert float(raw.grad) < 1e-3


def test_activation_preserves_temporal_signal_under_saturation() -> None:
    """Effect-on-Decision (Contract v4 §4): in a saturation regime, a recurrent-vs-memoryless shift in raw is
    SUPPRESSED by the pure tanh (logit delta ~ 0 -> behaviorally inert) but PRESERVED by the leaky map (logit
    delta > 0). Deterministic, isolates the activation as the single variable."""
    s, leak = 3.0, 0.1
    raw_mem = torch.tensor([20.0, 25.0, 30.0])              # same-sign saturation regime (|raw| >> s)
    raw_rec = raw_mem + torch.tensor([2.0, 3.0, 4.0])       # the recurrent hidden shifts raw

    z_tanh_mem, z_tanh_rec = s * torch.tanh(raw_mem / s), s * torch.tanh(raw_rec / s)
    z_leak_mem = z_tanh_mem + leak * raw_mem
    z_leak_rec = z_tanh_rec + leak * raw_rec

    d_tanh = recurrent_vs_memoryless_delta(z_tanh_rec, z_tanh_mem)
    d_leak = recurrent_vs_memoryless_delta(z_leak_rec, z_leak_mem)

    assert d_tanh["recurrent_memoryless_logit_delta"] < 0.05        # tanh saturates -> temporal signal lost
    assert d_leak["recurrent_memoryless_logit_delta"] > 0.15        # leak preserves it (~ leak*|raw_rec-raw_mem|)
    assert d_leak["recurrent_memoryless_logit_delta"] > 5.0 * d_tanh["recurrent_memoryless_logit_delta"]
