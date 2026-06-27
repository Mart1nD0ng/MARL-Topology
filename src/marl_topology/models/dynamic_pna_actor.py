"""D12 (Spec S7.12-7.13): the episode-recurrent, preference-conditioned, PNA directional dynamic actor.

A drop-in for ``DynamicRecurrentActor`` (identical ``forward(nf, ef, ei, hidden) -> (logits[E], h[N,H])``
contract) that upgrades the single mean-aggregation message-passing round to the full PNA readout
(4 aggregators x 3 degree scalers, Phase 11a) over DIRECTIONAL physical-neighbour messages, with a
SHARED cross-frame ``GRUCell`` carrying per-node state across episode frames. The deployment preference
``omega = (omega_E, omega_L)`` is a per-node input (a PUBLIC parameter -- deployment-legal, Spec S2.1),
so a SINGLE policy sweeps the energy-latency Pareto front (consumes the D11 preference primitive).

Fully DECENTRALIZED (D1): each node uses only its own (preference-augmented) features + its physical
in-neighbours' directed messages; no global state / global decoder / node ids. The activation is owned
by the torch-free local mutual-acceptance decoder (train == deploy). NaN-safe at isolated / single-
neighbour nodes (the gradient-safe std + degree-0 scaler mask live in the reused Phase-11 primitives).
``hidden=None`` removes the cross-frame memory (the memoryless arm) on the SAME architecture.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from marl_topology.models.pna_aggregation import PNA_SCALER_ALPHAS, pna_combine
from marl_topology.models.recurrent_directional_pna import scatter_directional_pna

DYNAMIC_PNA_ACTOR_MODEL_ID = "episode_recurrent_pna_directional_actor_v1"


class DynamicPNAActor(nn.Module):
    """PNA directional message passing + cross-frame GRU + preference conditioning -> per-edge logits."""

    model_id = DYNAMIC_PNA_ACTOR_MODEL_ID

    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 64, delta: float = 1.0,
                 alphas=PNA_SCALER_ALPHAS, pref_dim: int = 2):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden = hidden
        self.delta = float(delta)
        self.alphas = tuple(alphas)
        self.pref_dim = pref_dim
        self.node_enc = nn.Sequential(nn.Linear(node_dim + pref_dim, hidden), nn.ReLU())
        # directed-edge message from [source state, direction-specific edge features] (Spec S7.13)
        self.msg = nn.Sequential(nn.Linear(hidden + edge_dim, hidden), nn.ReLU())
        self.pna_proj = nn.Linear(4 * len(self.alphas) * hidden, hidden)
        self.gru = nn.GRUCell(hidden, hidden)                       # SHARED cross-frame recurrence
        self.h_norm = nn.LayerNorm(hidden)
        self.edge_head = nn.Sequential(nn.Linear(edge_dim + 2 * hidden, hidden), nn.ReLU(),
                                       nn.Linear(hidden, 1))
        self.logit_scale = 10.0
        self.register_buffer("omega", torch.zeros(pref_dim))

    def set_preference(self, *omega) -> "DynamicPNAActor":
        """Set the deployment preference ``omega = (omega_E, omega_L)`` (a public input). Returns self."""
        vals = list(omega) if len(omega) != 1 else list(omega[0])
        with torch.no_grad():
            self.omega.copy_(torch.tensor([float(o) for o in vals[: self.pref_dim]], device=self.omega.device))
        return self

    def init_hidden(self, n_nodes: int, ref: Tensor) -> Tensor:
        return ref.new_zeros(n_nodes, self.hidden)

    def forward(self, nf: Tensor, ef: Tensor, ei: Tensor,
                hidden: Tensor | None = None) -> tuple[Tensor, Tensor]:
        n = nf.shape[0]
        omega = self.omega.to(nf.dtype).unsqueeze(0).expand(n, self.pref_dim)
        x = self.node_enc(torch.cat([nf, omega], dim=-1))           # [N, H] preference-conditioned
        if ei.shape[0] > 0:
            u, v = ei[:, 0].long(), ei[:, 1].long()
            src = torch.cat([u, v], dim=0)                          # both directions (directional MP)
            dst = torch.cat([v, u], dim=0)
            ef2 = torch.cat([ef, ef], dim=0)
            m = self.msg(torch.cat([x[src], ef2], dim=-1))         # [2E, H] message from the SOURCE state
            agg = scatter_directional_pna(m, dst, n)               # [N, 4, H] (NaN-safe std)
            in_deg = x.new_zeros(n).scatter_add(0, dst, x.new_ones(dst.shape[0]))
            pna = pna_combine(agg, in_deg, self.delta, self.alphas)   # [N, 4*S*H] (degree-0 scaler masked)
            pna_h = self.pna_proj(pna)
        else:
            pna_h = x.new_zeros(n, self.hidden)
        gru_in = x + pna_h                                          # local + PNA neighbour summary
        h0 = hidden if hidden is not None else self.init_hidden(n, nf)
        h = self.gru(gru_in, h0)                                    # carries across frames
        if ei.shape[0] == 0:
            return ef.new_zeros(0), h
        hb = self.h_norm(h)
        hu, hv = hb[ei[:, 0].long()], hb[ei[:, 1].long()]
        feats = torch.cat([ef, hu * hv, torch.abs(hu - hv)], dim=-1)   # symmetric in (u, v)
        raw = self.edge_head(feats).squeeze(-1)
        logits = self.logit_scale * torch.tanh(raw / self.logit_scale)   # softly bounded, sign-preserving
        return logits, h

    def boundary_report(self) -> dict:
        return {
            "model_id": self.model_id,
            "global_topology_used": False,
            "decentralized_with_communication": True,
            "cross_frame_recurrence": True,
            "pna_aggregation": True,
            "preference_conditioned": True,
            "outputs": "per_edge_activation_logit",
            "activation_owned_by_decoder": True,
        }
