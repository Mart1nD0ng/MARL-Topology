"""Differentiable quorum-tail pooling -- a BFT-matched graph readout inductive bias.

PBFT consensus does not feasibly succeed when the AVERAGE node/link is good; it succeeds
when a QUORUM (>= 2f+1 of N) delivers. The generic masked-mean readout encodes the former;
this module encodes the latter. ``soft_quorum_tail`` is the differentiable counterpart of the
evaluator's ``heterogeneous_quorum_tail`` (same generating-polynomial DP), so a network that
pools through it carries the consensus order-statistic as an architectural prior instead of
having to learn it from a mean.

``QuorumTailReadout`` turns node embeddings into a per-node soft readiness gate, then reports
the quorum-tail probability around the operating quorum k(N) = 2*floor((N-1)/3)+1 -- the same
fault-tolerant quorum the PBFT model uses -- plus a gate-weighted embedding pool so
representational capacity is preserved. Pure function of the (gates, mask, quorum); no scene
or evaluator access, so it stays a drop-in pooling layer.
"""

from __future__ import annotations

import torch
from torch import nn

# tail at (k-1, k, k+1) + expected ready fraction
QUORUM_TAIL_FEATURE_DIM = 4


def soft_quorum_tail(
    gates: torch.Tensor,
    mask: torch.Tensor,
    quorum: torch.Tensor | int,
) -> torch.Tensor:
    """Differentiable P(at least ``quorum`` of the masked Bernoulli(gates_i) succeed).

    Vectorized Poisson-binomial DP over the generating polynomial prod_i((1-g_i) + g_i z);
    masked entries get g_i = 0 so they never contribute a success (DP-invariant). Matches
    ``heterogeneous_quorum_tail`` to floating precision on hard (0/1-grad) inputs.

    gates, mask: [B, N]; quorum: scalar int or [B] long. Returns [B].
    """
    if gates.shape != mask.shape:
        raise ValueError("gates and mask must share shape [B, N]")
    batch, n = gates.shape
    g = gates * mask
    dp = torch.zeros(batch, n + 1, device=gates.device, dtype=gates.dtype)
    dp[:, 0] = 1.0
    zero_col = torch.zeros(batch, 1, device=gates.device, dtype=gates.dtype)
    for i in range(n):
        g_i = g[:, i : i + 1]
        shifted = torch.cat((zero_col, dp[:, :-1]), dim=1)
        dp = dp * (1.0 - g_i) + shifted * g_i
    degrees = torch.arange(n + 1, device=gates.device).unsqueeze(0)
    if torch.is_tensor(quorum):
        tail_mask = (degrees >= quorum.unsqueeze(1)).to(dp.dtype)
    else:
        tail_mask = (degrees >= int(quorum)).to(dp.dtype)
    return (dp * tail_mask).sum(dim=1)


def operating_quorum(n_eff: torch.Tensor, fault_cap: int | None = 1) -> torch.Tensor:
    """PBFT commit quorum k = 2f+1, f = floor((N-1)/3) capped at ``fault_cap``, per element.

    ``fault_cap=1`` matches the project evaluator (Stage 21 caps fault_tolerance at 1, so
    the commit quorum is 3 for every N >= 4); ``fault_cap=None`` is the textbook scaling
    quorum 2*floor((N-1)/3)+1."""
    fault = torch.clamp((n_eff - 1.0) / 3.0, min=0.0).floor()
    if fault_cap is not None:
        fault = torch.clamp(fault, max=float(fault_cap))
    return (2.0 * fault + 1.0).long()


class QuorumTailReadout(nn.Module):
    """BFT-matched graph readout: a per-node soft readiness gate pooled through a SPREAD of
    consensus order-statistics -- the low commit quorum (2f+1) AND high near-unanimity
    (N-1, N). The spread matters because the evaluator AVERAGES expected-initiator consensus
    over all N primaries, so a single non-ready node caps feasibility at (N-1)/N: the dominant
    feasibility signal lives at the high-quorum (near-all-ready) end, while the low commit
    quorum captures fault tolerance. Features [B, hidden+4] = (gate-weighted pool, [tail@commit
    quorum, tail@N-1, tail@N, expected ready fraction])."""

    def __init__(self, hidden_dim: int, fault_cap: int | None = 1) -> None:
        super().__init__()
        self.gate_head = nn.Linear(hidden_dim, 1)
        self.fault_cap = fault_cap

    def forward(
        self,
        node_state: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gates = torch.sigmoid(self.gate_head(node_state).squeeze(-1)) * node_mask  # [B, N]
        n_eff = node_mask.sum(dim=1)
        commit_quorum = operating_quorum(n_eff, self.fault_cap)
        near_all = torch.clamp(n_eff - 1.0, min=1.0).long()
        all_ready = torch.clamp(n_eff, min=1.0).long()
        tail_commit = soft_quorum_tail(gates, node_mask, commit_quorum)
        tail_near_all = soft_quorum_tail(gates, node_mask, near_all)
        tail_all = soft_quorum_tail(gates, node_mask, all_ready)
        expected_ready = gates.sum(dim=1) / n_eff.clamp_min(1.0)
        tail_features = torch.stack(
            (tail_commit, tail_near_all, tail_all, expected_ready), dim=-1
        )
        denominator = gates.sum(dim=1, keepdim=True).clamp_min(1e-6)
        weighted_pool = (gates.unsqueeze(-1) * node_state).sum(dim=1) / denominator
        return weighted_pool, tail_features
