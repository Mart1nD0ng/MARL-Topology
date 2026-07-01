"""T5 (Temporal Recovery) — heteroscedastic uncertainty head + confidence-gated correction (task 4.2/4.3).

T1-T4 showed the correction captures direction but not magnitude, and applying the (noisy) magnitude everywhere
HURTS the anchor. T5 predicts the correction MEAN mu AND a per-edge log-variance logvar (Gaussian NLL), and at
inference applies the correction ONLY where confident (gate = sigmoid(-logvar)), else falls back to the stale
value (echo). Opt-in ``belief_uncertainty`` (default False -> byte-identical). Fails on HEAD: no belief_uncertainty
/ belief_logvar_head / belief_with_uncertainty, and no csi_correction_nll / confidence_gate.
"""

from __future__ import annotations

import torch

from marl_topology.models.belief_residual_actor import BeliefResidualActor
from marl_topology.training.csi_belief import (confidence_gate, csi_correction_nll, gated_correction,
                                               uncertainty_calibration)


def test_belief_uncertainty_adds_logvar_head() -> None:
    a0 = BeliefResidualActor(4, 6, hidden=8)
    assert not hasattr(a0, "belief_logvar_head")
    a1 = BeliefResidualActor(4, 6, hidden=8, belief_uncertainty=True)
    assert isinstance(a1.belief_logvar_head, torch.nn.Sequential)
    assert a1.boundary_report()["belief_uncertainty"] is True


def test_belief_with_uncertainty_mu_equals_belief() -> None:
    """mu (the correction mean) is identical to belief() -- same head + features; logvar is the extra output."""
    a = BeliefResidualActor(4, 6, hidden=8, belief_extra_dim=0, belief_uncertainty=True)
    nf = torch.randn(5, 4)
    ei = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
    ef = torch.rand(3, 6)
    bel, _h = a.belief(nf, ef, ei)
    mu, logvar, _h2 = a.belief_with_uncertainty(nf, ef, ei)
    assert torch.allclose(bel, mu)
    assert logvar.shape == (3,)


def test_confidence_gate_shrinks_uncertain_corrections() -> None:
    assert float(confidence_gate(torch.tensor([6.0]))) < 0.01        # high variance -> gate ~ 0
    assert float(confidence_gate(torch.tensor([-6.0]))) > 0.99       # low variance  -> gate ~ 1
    mu = torch.tensor([1.0, 1.0])
    lv = torch.tensor([6.0, -6.0])                                   # edge 0 uncertain, edge 1 confident
    gc = gated_correction(mu, lv)
    assert float(gc[0].abs()) < 0.01 and float(gc[1]) > 0.99         # shrink uncertain, keep confident


def test_nll_is_heteroscedastic() -> None:
    """A higher logvar REDUCES the NLL penalty on a large error (the variance absorbs it) -- so mu is not forced
    to shrink toward the mean the way MSE forces it."""
    mu, tgt = torch.tensor([0.0]), torch.tensor([2.0])              # large residual
    nll_lowvar = csi_correction_nll(mu, torch.tensor([0.0]), tgt)
    nll_highvar = csi_correction_nll(mu, torch.tensor([2.0]), tgt)
    assert float(nll_highvar) < float(nll_lowvar)


def test_uncertainty_calibration_positive_when_logvar_tracks_error() -> None:
    mu = torch.zeros(4)
    tgt = torch.tensor([0.1, 2.0, 0.1, 2.0])                        # errors: small, large, small, large
    logvar = torch.tensor([-3.0, 3.0, -3.0, 3.0])                   # logvar tracks the error
    assert uncertainty_calibration(mu, logvar, tgt) > 0.5


def test_belief_uncertainty_false_byte_identical() -> None:
    """belief_uncertainty=False (default) is byte-identical: every shared param preserved, only the logvar head
    added when enabled (appended last)."""
    torch.manual_seed(0)
    a0 = BeliefResidualActor(4, 6, hidden=8, belief_extra_dim=2)
    torch.manual_seed(0)
    a1 = BeliefResidualActor(4, 6, hidden=8, belief_extra_dim=2, belief_uncertainty=True)
    sd0, sd1 = a0.state_dict(), a1.state_dict()
    for k, v in sd0.items():
        assert k in sd1 and torch.equal(v, sd1[k])
    extra = set(sd1) - set(sd0)
    assert extra and all("logvar" in k for k in extra)
