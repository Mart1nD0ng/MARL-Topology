"""R1 (Belief-Guided Residual PPO) — actor with a SEPARATE small-range residual head (logit-saturation fix).

Q14 root cause: the residual path reused the shared +-10 tanh activation head, whose trained ``raw`` ~ 3000
saturated 83% of logits and annihilated the GRU's cross-frame contribution -> recurrent == memoryless. This
actor keeps the PROVEN node-encoder + directional message passing + per-node GRUCell (the cross-frame
recurrence of ``DynamicRecurrentActor``), but routes the residual decision through a DEDICATED head that
outputs SMALL-range logits ``z = z_max * tanh(raw / z_max)`` with ``z_max in [2,4]`` (default 3) and EXPOSES
``raw`` for the raw-logit L2 penalty + saturation metrics (R1) -> the temporal signal can reach the logits.

Belief / repair / safety heads are placeholders for R2 / R5; this stage only adds the residual policy head.
Fully decentralized (local node features + physical-neighbour messages only); no global state / decoder.
``forward(nf, ef, ei, hidden) -> (residual_logits[E], raw[E], new_hidden[N,H])``; ``hidden=None`` => the GRU
starts from zeros (the memoryless ablation), so the two arms differ ONLY in whether the hidden carries.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

BELIEF_RESIDUAL_ACTOR_MODEL_ID = "belief_residual_actor_v1"


class BeliefResidualActor(nn.Module):
    """GRU encoder (cross-frame hidden) + a separate small-range residual policy head exposing ``raw``."""

    model_id = BELIEF_RESIDUAL_ACTOR_MODEL_ID

    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 64,
                 residual_logit_scale: float = 3.0, belief_extra_dim: int = 0) -> None:
        super().__init__()
        if not (2.0 <= residual_logit_scale <= 4.0):
            raise ValueError(f"residual_logit_scale should be in [2,4] (got {residual_logit_scale})")
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden = hidden
        self.residual_logit_scale = float(residual_logit_scale)
        self.belief_extra_dim = int(belief_extra_dim)   # R2: leak-free per-edge extras (velocity) for belief
        self.node_enc = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU())
        self.msg = nn.Sequential(nn.Linear(2 * hidden + edge_dim, hidden), nn.ReLU())
        self.gru = nn.GRUCell(hidden, hidden)                       # CROSS-FRAME recurrence
        self.h_norm = nn.LayerNorm(hidden)
        # SEPARATE residual policy head (NOT the shared +-10 activation head) -> small-range residual logit
        self.residual_head = nn.Sequential(nn.Linear(edge_dim + 2 * hidden, hidden), nn.ReLU(),
                                            nn.Linear(hidden, 1))
        # R2: CSI belief head -> per-edge predicted CURRENT link-psucc LOGIT (supervised by L_CSI, training-
        # only true-CSI label). Appended LAST so the R1 residual_head init RNG is unchanged. ``belief_extra_
        # dim`` admits leak-free per-edge extras (relative velocity / distance_delta) the head needs to
        # extrapolate the current channel from the stale observation.
        self.belief_head = nn.Sequential(
            nn.Linear(edge_dim + self.belief_extra_dim + 2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def init_hidden(self, n_nodes: int, ref: Tensor) -> Tensor:
        return ref.new_zeros(n_nodes, self.hidden)

    def encode(self, nf: Tensor, ef: Tensor, ei: Tensor, hidden: Tensor | None):
        """node encode -> one directional message-passing round -> GRUCell update -> bounded node embedding."""
        n = nf.shape[0]
        x = self.node_enc(nf)
        agg = x.new_zeros(n, self.hidden)
        if ei.shape[0] > 0:
            u, v = ei[:, 0].long(), ei[:, 1].long()
            src = torch.cat([u, v], dim=0)
            dst = torch.cat([v, u], dim=0)
            ef2 = torch.cat([ef, ef], dim=0)
            m = self.msg(torch.cat([x[src], x[dst], ef2], dim=-1))
            agg = agg.index_add(0, dst, m)
            deg = agg.new_zeros(n).index_add(0, dst, torch.ones(dst.shape[0], device=agg.device))
            agg = agg / deg.clamp_min(1.0).unsqueeze(-1)
        h0 = hidden if hidden is not None else self.init_hidden(n, nf)
        h = self.gru(x + agg, h0)
        return h

    def residual_raw(self, ef: Tensor, ei: Tensor, h: Tensor) -> Tensor:
        hb = self.h_norm(h)
        hu, hv = hb[ei[:, 0].long()], hb[ei[:, 1].long()]
        feats = torch.cat([ef, hu * hv, torch.abs(hu - hv)], dim=-1)     # symmetric in (u, v)
        return self.residual_head(feats).squeeze(-1)

    def forward(self, nf: Tensor, ef: Tensor, ei: Tensor,
                hidden: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        h = self.encode(nf, ef, ei, hidden)
        if ei.shape[0] == 0:
            return ef.new_zeros(0), ef.new_zeros(0), h
        raw = self.residual_raw(ef, ei, h)
        s = self.residual_logit_scale
        residual_logits = s * torch.tanh(raw / s)          # small-range, sign- and order-preserving
        return residual_logits, raw, h

    def belief(self, nf: Tensor, ef: Tensor, ei: Tensor, extra: Tensor | None = None,
               hidden: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """R2 CSI belief: per-edge predicted CURRENT-psucc LOGIT from the GRU hidden (cross-frame). Shares
        the encoder/GRU with the residual head, so L_CSI gradients flow into the GRU. ``extra`` is optional
        LEAK-FREE per-edge features (e.g. relative velocity / distance_delta) of width ``belief_extra_dim``.
        Returns (belief_logit[E], new_hidden[N,H]). The deployed actor NEVER reads the true current CSI."""
        h = self.encode(nf, ef, ei, hidden)
        if ei.shape[0] == 0:
            return ef.new_zeros(0), h
        hb = self.h_norm(h)
        hu, hv = hb[ei[:, 0].long()], hb[ei[:, 1].long()]
        parts = [ef]
        if self.belief_extra_dim:
            if extra is None:
                extra = ef.new_zeros(ef.shape[0], self.belief_extra_dim)
            parts.append(extra)
        parts += [hu * hv, torch.abs(hu - hv)]
        bel = self.belief_head(torch.cat(parts, dim=-1)).squeeze(-1)
        return bel, h

    def boundary_report(self) -> dict:
        return {
            "model_id": self.model_id,
            "global_topology_used": False,
            "decentralized_with_communication": True,
            "cross_frame_recurrence": True,
            "separate_residual_head": True,
            "residual_logit_scale": self.residual_logit_scale,
            "outputs": "small_range_residual_logit + raw",
        }
