"""T0 (Temporal Recovery) — audit tripwires pinning the 4 chain-1 defects at HEAD.

These are LOAD-BEARING regression tripwires (Contract v4 §3/§4): each asserts a concrete decision-path fact
about the CURRENT (to-be-fixed) architecture. They pass on HEAD and are designed to FLIP when the campaign
fixes each defect, so a fix cannot land silently:
  * test_belief_target_is_absolute_current_echo_optimal  -> flips at T3 (correction target)
  * test_residual_logit_saturates_and_gradient_vanishes  -> flips at T2 (non-saturating activation)
  * test_recurrence_is_per_node_and_no_edge_state         -> flips at T4 (edge-level recurrent state)
They document WHY the prior belief/temporal chain failed structurally (not because recovery is impossible).
"""

from __future__ import annotations

import torch

from marl_topology.models.belief_residual_actor import BeliefResidualActor
from marl_topology.training.csi_belief import _logit, belief_target_logits, csi_belief_loss


class _Rec:
    def __init__(self, p: float) -> None:
        self.link_success_probability = p


class _Ctx:
    def __init__(self, recs: dict) -> None:
        self.link_records = recs


class _Scene:
    """Minimal stub exposing context(t).link_records[eid].link_success_probability (the true current channel)."""

    def __init__(self, recs: dict) -> None:
        self._recs = recs

    def context(self, t: int) -> _Ctx:
        return _Ctx(self._recs)


def test_belief_target_is_absolute_current_echo_optimal() -> None:
    """DEFECT 1 (task 3.1): the belief target is the ABSOLUTE current psucc logit, not a stale->current
    correction. Consequence: for a stale==current edge the loss-optimal output is logit(stale) -> echoing the
    observation is a global optimum, so the head has no incentive to recover anything. Flips at T3."""
    true_p = 0.83
    scene = _Scene({7: _Rec(true_p)})
    edge_ids = [7]
    target = belief_target_logits(scene, 0, edge_ids)

    # (a) the target is the ABSOLUTE current logit -- it does not depend on any stale observation (not a delta).
    assert torch.allclose(target, _logit(torch.tensor([true_p])).float(), atol=1e-5)

    # (b) for a stale==current edge, echoing the stale input's logit yields ZERO loss -> echo is loss-optimal.
    stale_input = true_p  # stale observation happens to equal the current channel
    echo_logit = _logit(torch.tensor([stale_input])).float()
    loss_echo = csi_belief_loss(echo_logit, target)
    assert float(loss_echo) < 1e-8


def test_residual_logit_saturates_and_gradient_vanishes() -> None:
    """DEFECT 2 (task 2 / 3.3): the acted residual logit is z = z_max*tanh(raw/z_max). It saturates to +-z_max
    and its gradient vanishes for |raw| >> z_max, so the recurrent contribution to `raw` cannot reach the acted
    logit once raw grows. Flips at T2 (non-saturating map)."""
    torch.manual_seed(0)
    actor = BeliefResidualActor(node_dim=4, edge_dim=6, hidden=8, residual_logit_scale=3.0)
    s = actor.residual_logit_scale

    # (a) the actor's forward genuinely applies the saturating map (load-bearing on the decision path).
    nf = torch.randn(5, 4)
    ei = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
    ef = torch.randn(3, 6)
    residual_logits, raw, _h = actor.forward(nf, ef, ei)
    assert torch.allclose(residual_logits, s * torch.tanh(raw / s), atol=1e-6)

    # (b) rails to +-s and the gradient vanishes at the rail.
    big = torch.tensor([100.0], requires_grad=True)
    out = s * torch.tanh(big / s)
    out.backward()
    assert abs(float(out) - s) < 1e-3          # railed to +s
    assert float(big.grad) < 1e-3              # d(logit)/d(raw) -> 0 at saturation
    neg = s * torch.tanh(torch.tensor([-100.0]) / s)
    assert abs(float(neg) + s) < 1e-3          # railed to -s


def test_recurrence_is_per_node_and_no_edge_state() -> None:
    """DEFECT 3+4 (task 3.4 / 4.1): the only recurrent cell is a per-node GRUCell (hidden [N,H]) while the CSI
    dynamics it must track are per-edge (ef rows); and there is NO edge-level recurrent state. Flips at T4."""
    actor = BeliefResidualActor(node_dim=4, edge_dim=6, hidden=8)

    recurrent = [m for m in actor.modules()
                 if isinstance(m, (torch.nn.GRUCell, torch.nn.LSTMCell, torch.nn.GRU, torch.nn.LSTM))]
    assert len(recurrent) == 1                 # exactly one recurrent module today
    assert isinstance(recurrent[0], torch.nn.GRUCell)
    assert isinstance(actor.gru, torch.nn.GRUCell)

    n_nodes = 5
    nf = torch.randn(n_nodes, 4)
    ei = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)   # 3 edges != 5 nodes
    ef = torch.randn(3, 6)
    _rl, _raw, h = actor.forward(nf, ef, ei)
    assert h.shape == (n_nodes, 8)             # hidden is per-NODE [N,H]
    assert h.shape[0] != ei.shape[0]           # NOT per-edge

    # no edge-level recurrent state / hidden exists yet (tripwire; flips when T4 adds it)
    assert not hasattr(actor, "edge_gru")
    assert not hasattr(actor, "edge_lstm")
    assert not hasattr(actor, "edge_hidden")
