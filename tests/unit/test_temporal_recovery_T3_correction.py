"""T3 (Temporal Recovery) — the belief CORRECTION target (task 3.1/3.2).

R2's belief head predicted the ABSOLUTE current psucc logit and echoed the stale input (never beat the
stale-echo floor). T3 reparametrizes: belief_logit = stale_logit + head_output, so the head predicts the
SIGNED stale->current correction and echo (head=0) is the zero-baseline -- stale-echo is no longer a global
optimum. These tests pin the correction primitives + the generator's parametrization path (load-bearing).
Fails on HEAD: csi_belief has no correction primitives and the T3 generator does not exist yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from marl_topology.models.belief_residual_actor import BeliefResidualActor
from marl_topology.training.csi_belief import (_logit, belief_correction_target, directional_accuracy,
                                               stale_logit)

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT / "scripts" / "train", _ROOT / "scripts" / "diagnostics"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


class _Rec:
    def __init__(self, p: float) -> None:
        self.link_success_probability = p


class _Ctx:
    def __init__(self, recs: dict) -> None:
        self.link_records = recs


class _Scene:
    def __init__(self, recs: dict) -> None:
        self._recs = recs

    def context(self, t: int) -> _Ctx:
        return _Ctx(self._recs)


def test_correction_target_makes_echo_nonoptimal() -> None:
    """For a MOVED edge (true != stale) the correction target is nonzero, so echo (correction 0) is NOT
    loss-optimal (fixes R2's stale-echo optimum); where the channel did not move it is ~0 (echo IS correct)."""
    stale = torch.tensor([0.6])
    ct_moved = belief_correction_target(_Scene({7: _Rec(0.9)}), 0, [7], stale)    # true 0.9 vs stale 0.6
    exp = _logit(torch.tensor([0.9])).float() - stale_logit(stale)
    assert torch.allclose(ct_moved, exp, atol=1e-5)
    assert float(ct_moved.abs()) > 0.1                                            # echo (0) is NOT optimal
    ct_still = belief_correction_target(_Scene({7: _Rec(0.6)}), 0, [7], stale)    # true == stale
    assert float(ct_still.abs()) < 1e-4                                           # echo IS optimal here


def test_correction_echo_zero_baseline_reproduces_stale() -> None:
    """head_output=0 -> belief = stale_logit -> sigmoid = stale psucc exactly (echo is the zero-baseline)."""
    stale = torch.tensor([0.7, 0.4, 0.9])
    echo = stale_logit(stale) + torch.zeros_like(stale)
    assert torch.allclose(torch.sigmoid(echo), stale, atol=1e-4)


def test_directional_accuracy_rewards_correct_sign() -> None:
    """directional_accuracy = frac of MOVED edges with correct correction sign (task 3.2)."""
    ct = torch.tensor([0.5, -0.5, 0.3, -0.4])
    assert directional_accuracy(ct.clone(), ct) == 1.0          # perfect
    assert directional_accuracy(-ct, ct) == 0.0                 # inverted
    assert directional_accuracy(torch.zeros_like(ct), ct) == 0.0  # echo -> no directional signal


def test_generator_parametrization_is_single_variable() -> None:
    """Load-bearing on the T3 generator: absolute uses belief_logit=head; correction uses stale_logit+head;
    same actor + input -> identical head, so the ONLY difference is the +stale_logit term."""
    import t3_correction_belief_gen as g3
    nd, ed = 4, 6
    actor = BeliefResidualActor(nd, ed, hidden=8, belief_extra_dim=0, residual_leak=0.1)
    obs = {"nf": torch.randn(4, nd), "ef": torch.rand(3, ed),
           "ei": torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)}
    mean = (torch.zeros(nd), torch.zeros(ed))
    std = (torch.ones(nd), torch.ones(ed))
    bl_abs, head_abs, _ = g3._belief_logit(actor, obs, mean, std, None, None, "absolute")
    bl_cor, head_cor, _ = g3._belief_logit(actor, obs, mean, std, None, None, "correction")
    assert torch.allclose(head_abs, head_cor)                      # same head (single variable)
    assert torch.allclose(bl_abs, head_abs)                        # absolute = head
    assert torch.allclose(bl_cor, stale_logit(obs["ef"][:, 0]) + head_cor)   # correction = stale_logit + head
    assert not torch.allclose(bl_abs, bl_cor)
