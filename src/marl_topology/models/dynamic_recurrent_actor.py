"""Episode-recurrent decentralized actor (Spec S7.12 temporal actor, two-timescale rollout).

Carries a per-node hidden state ACROSS episode frames -- the cross-frame recurrence the static
T=1 actors lack (their GRU runs only across message-passing rounds within one forward). Each frame:

    local node encode -> one directional message-passing round (physical neighbours) -> GRUCell
    update of the per-node hidden state carried from the previous frame -> SYMMETRIC edge head ->
    per-candidate-edge activation logits [E]  (-> BCSP sampler -> local_mutual_assemble, downstream).

``forward(nf, ef, ei, hidden) -> (logits[E], new_hidden[N,H])``. With ``hidden=None`` the GRU starts
from zeros -> the cross-frame memory is REMOVED, and the SAME architecture (identical parameters /
capacity) becomes a per-frame reactive policy. The dynamic rollout uses this to run two arms that
differ ONLY in whether the hidden state carries: a controlled ablation of cross-frame memory.

Fully DECENTRALIZED (D1): per-node local features + physical-neighbour messages only; no global
state / global decoder / node ids; device-preserving; variable N / E. The activation is owned by the
torch-free local mutual-acceptance decoder, identical to deploy (train == deploy).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

DYNAMIC_RECURRENT_ACTOR_MODEL_ID = "episode_recurrent_directional_actor_v1"


class DynamicRecurrentActor(nn.Module):
    """Per-node hidden state carried across episode frames -> per-candidate-edge activation logits."""

    model_id = DYNAMIC_RECURRENT_ACTOR_MODEL_ID

    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 64):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden = hidden
        self.node_enc = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU())
        # directed message: [h[src], h[dst], edge_feat] -> hidden
        self.msg = nn.Sequential(nn.Linear(2 * hidden + edge_dim, hidden), nn.ReLU())
        self.gru = nn.GRUCell(hidden, hidden)                       # CROSS-FRAME recurrence
        # LayerNorm bounds the per-node embedding (like the static actor's node/edge norms) so the
        # recurrent BPTT cannot blow the hidden state up into runaway logits / a NaN sampler.
        self.h_norm = nn.LayerNorm(hidden)
        # SYMMETRIC edge head: [edge_feat, h[u]*h[v], |h[u]-h[v]|] -> scalar logit (undirected)
        self.edge_head = nn.Sequential(nn.Linear(edge_dim + 2 * hidden, hidden), nn.ReLU(),
                                       nn.Linear(hidden, 1))
        # the per-edge activation logit is softly bounded to (-LOGIT_SCALE, LOGIT_SCALE): small logits
        # stay ~linear, large ones saturate -> exp(theta) in the BCSP sampler can never overflow. The
        # sign (the logit>=0 gate) and relative order are preserved -> decoder semantics unchanged.
        self.logit_scale = 10.0

    def init_hidden(self, n_nodes: int, ref: Tensor) -> Tensor:
        return ref.new_zeros(n_nodes, self.hidden)

    def forward(self, nf: Tensor, ef: Tensor, ei: Tensor,
                hidden: Tensor | None = None) -> tuple[Tensor, Tensor]:
        n = nf.shape[0]
        x = self.node_enc(nf)                                       # [N, H]
        agg = x.new_zeros(n, self.hidden)
        if ei.shape[0] > 0:
            u, v = ei[:, 0].long(), ei[:, 1].long()
            # both directions (directional message passing); aggregate incoming by MEAN
            src = torch.cat([u, v], dim=0)
            dst = torch.cat([v, u], dim=0)
            ef2 = torch.cat([ef, ef], dim=0)
            m = self.msg(torch.cat([x[src], x[dst], ef2], dim=-1))  # [2E, H]
            agg = agg.index_add(0, dst, m)
            deg = agg.new_zeros(n).index_add(0, dst, torch.ones(dst.shape[0], device=agg.device))
            agg = agg / deg.clamp_min(1.0).unsqueeze(-1)
        gru_in = x + agg                                            # local + neighbour summary
        h0 = hidden if hidden is not None else self.init_hidden(n, nf)
        h = self.gru(gru_in, h0)                                    # [N, H]  (carries across frames)
        if ei.shape[0] == 0:
            return ef.new_zeros(0), h
        hb = self.h_norm(h)                                         # bounded embedding for the edge head
        hu, hv = hb[ei[:, 0].long()], hb[ei[:, 1].long()]
        feats = torch.cat([ef, hu * hv, torch.abs(hu - hv)], dim=-1)   # symmetric in (u, v)
        raw = self.edge_head(feats).squeeze(-1)
        logits = self.logit_scale * torch.tanh(raw / self.logit_scale)   # softly bounded, sign-preserving
        return logits, h                                            # [E], [N, H]

    def boundary_report(self) -> dict:
        return {
            "model_id": self.model_id,
            "global_topology_used": False,
            "decentralized_with_communication": True,
            "cross_frame_recurrence": True,
            "outputs": "per_edge_activation_logit",
            "activation_owned_by_decoder": True,
        }
