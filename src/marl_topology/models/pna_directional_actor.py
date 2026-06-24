"""Phase 11c (Technical-Spec S7.12): the deployable Preference-conditioned Directional PNA Actor.

Assembles the Spec S7.12 actor stack on the Phase-11a/11b primitives:
  node encoder (+ omega preference) -> directional message passing -> PNA aggregation -> recurrent
  shared update (RecurrentDirectionalPNA backbone) -> SYMMETRIC directed-bid edge-logit head.

It is a DROP-IN for ``MessagePassingGraphEdgeScorer``: same ``forward(node_features, edge_features,
edge_index, node_mask, edge_mask) -> [B, E]`` per-candidate-edge activation logits, so ``forward_logits``
+ the BCSP sampler + the local mutual-acceptance decoder are reused UNCHANGED. The extra ``omega``
keyword (preference ``(omega_E, omega_L)``, Spec S6.1) is concatenated to the node features so ONE policy
spans the energy-latency Pareto front; it defaults to a neutral ``(0.5, 0.5)`` so the 5-arg call from
``forward_logits`` works as-is.

Each undirected candidate edge is message-passed in BOTH directions (directional backbone) but its
activation logit is SYMMETRIC in its endpoints (the activation is mutual, decided by both endpoints'
BCSP). Fully DECENTRALIZED (D1): per-node local features + physical-neighbour messages + public protocol
params only; no global state / global decoder / node IDs; device-preserving; variable N/E.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from marl_topology.models.recurrent_directional_pna import RecurrentDirectionalPNA

PNA_DIRECTIONAL_ACTOR_MODEL_ID = "preference_conditioned_directional_pna_actor_v1"
_OMEGA_DIM = 2   # (omega_E, omega_L)


class PNADirectionalActor(nn.Module):
    """Preference-conditioned directional PNA actor -> per-candidate-edge activation logits ``[B, E]``."""

    model_id = PNA_DIRECTIONAL_ACTOR_MODEL_ID

    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 64, rounds: int = 3,
                 delta: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden = hidden
        # the backbone sees the node features PLUS the omega preference (concatenated per node)
        self.backbone = RecurrentDirectionalPNA(node_dim + _OMEGA_DIM, edge_dim, hidden=hidden,
                                                rounds=rounds, delta=delta)
        # SYMMETRIC edge head: [edge_feat, H[u]*H[v], |H[u]-H[v]|] -> scalar logit (undirected activation)
        self.edge_head = nn.Sequential(nn.Linear(edge_dim + 2 * hidden, hidden), nn.ReLU(),
                                       nn.Dropout(dropout), nn.Linear(hidden, 1))

    def _neutral_omega(self, ref: Tensor) -> Tensor:
        return ref.new_full((_OMEGA_DIM,), 0.5)

    def _forward_single(self, nf: Tensor, ef: Tensor, ei: Tensor, omega: Tensor) -> Tensor:
        n = nf.shape[0]
        nf_w = torch.cat([nf, omega.to(nf.dtype).expand(n, _OMEGA_DIM)], dim=-1)   # preference per node
        if ei.shape[0] == 0:
            h = self.backbone(nf_w, ei.new_zeros(0, 2), ef.new_zeros(0, self.edge_dim))
            return ef.new_zeros(0)
        directed = torch.cat([ei, ei[:, [1, 0]]], dim=0)            # both directions (directional backbone)
        directed_ef = torch.cat([ef, ef], dim=0)
        h = self.backbone(nf_w, directed, directed_ef)              # [N, hidden]
        u, v = ei[:, 0].long(), ei[:, 1].long()
        hu, hv = h[u], h[v]
        feats = torch.cat([ef, hu * hv, torch.abs(hu - hv)], dim=-1)  # symmetric in (u, v)
        return self.edge_head(feats).squeeze(-1)                    # [E]

    def forward(self, node_features, edge_features, edge_index, node_mask, edge_mask, omega=None):
        # signature-compatible with MessagePassingGraphEdgeScorer; node_mask/edge_mask are accepted for
        # interface parity (forward_logits passes all-ones, B=1). Batched by looping (B=1 in practice).
        b = node_features.shape[0]
        om = omega if omega is not None else self._neutral_omega(node_features)
        if om.dim() == 1:
            om = om.unsqueeze(0).expand(b, _OMEGA_DIM)              # broadcast one preference to the batch
        outs = [self._forward_single(node_features[i], edge_features[i], edge_index[i], om[i])
                for i in range(b)]
        return torch.stack(outs, dim=0)                            # [B, E]

    def boundary_report(self) -> dict:
        """Deployment boundary (D1): directional K-hop local message passing, omega-conditioned, no
        global state / global decoder; outputs per-edge activation logits owned by the decoder."""
        return {
            "model_id": self.model_id,
            "global_topology_used": False,
            "receptive_field_hops": self.backbone.rounds,
            "decentralized_with_communication": True,
            "preference_conditioned": True,
            "outputs": "per_edge_activation_logit",
            "activation_owned_by_decoder": True,
        }
