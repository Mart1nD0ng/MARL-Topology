"""Local K-hop message-passing GNN edge scorer (decentralized-with-communication).

Design intent (why this exists alongside the v3 ego-graph scorer)
-----------------------------------------------------------------
The Stage-33 v3 actor (``local_gnn_edge_scorer.py``) groups candidate edges into
**one ego node per group** and reduces every incident edge of that ego into a single
ego vector. Its receptive field is therefore exactly **one hop** (the ego's own
incident edges): it can never reason about a 2- or 3-hop backbone such as "keep the
weak link v0--v1 because v1--rsu0 carries the relay that v0 needs for quorum". That
1-hop ceiling is a structural cause of the v3 training collapse.

This module is the honest fix. It runs **K rounds of message passing over the real
candidate-graph adjacency reconstructed from actor-safe rows**. Each round is one hop
of physical neighbour-to-neighbour signalling (realistic V2V / V2I exchange): a node
sends its current state to its neighbours, and after round ``k`` a node's state already
summarises its ``k``-hop neighbourhood. After ``K`` rounds the logit for a directed
edge ``(u, v)`` depends on the ``K``-hop subgraph around ``u`` and ``v`` -- and on
**nothing outside it**. The actor never reads the global adjacency as a feature; it
only ever sees its own incident edges per round. This is the standard GNN-as-
communication view of a Dec-POMDP actor (decentralised, *not* the free-global-state
shortcut), and it is verified by a receptive-field test (a perturbation ``d`` hops away
is invisible until ``K >= d``) and a batch-locality test (no cross-scene leakage).

Directed scoring
----------------
The head is **directed**: node ``u``'s preference for edge ``(u, v)`` is
``score(edge_uv, h_u, h_v)`` with ``u`` in the "self" slot, which differs from ``v``'s
``score(edge_vu, h_v, h_u)``. This is exactly what per-node *mutual acceptance* needs
(each directed accept is owned by exactly one agent), and it maps 1:1 onto the existing
directed ``EdgeScoreRecord`` plumbing, so downstream aggregation/decoding is unchanged.

Anti-oversmoothing (mandatory at ``K >= 3``)
--------------------------------------------
Per-node ``LayerNorm`` (row-wise -> does not couple nodes, so strict locality holds),
residual connections, and Jumping-Knowledge over per-round edge states. An optional
per-scene centering (PairNorm-lite) is available as an ablation but defaults OFF because
subtracting a scene mean weakly couples disconnected components. The final score head is
initialised at a small symmetric scale (mean edge probability ~ 0.5) reusing the v3
``output_head_init_gain`` lesson, so the policy starts unsaturated yet committed.

Deployment boundary: actor-safe local features only. The hard topology activation
remains owned by the environment-side decoder/assembler; this module outputs directed
edge scores only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .tensorizers import (
    ACTOR_EDGE_FEATURE_FIELDS,
    ACTOR_NODE_FEATURE_FIELDS,
    LocalGraphTensorBatch,
)


LOCAL_KHOP_GNN_EDGE_SCORER_V1_MODEL_ID = "local_khop_message_passing_gnn_edge_scorer_v1"

# Same forbidden-input contract the v3 actor enforces; this model registers its own copy
# so the boundary tests can validate it independently.
KHOP_GNN_FORBIDDEN_INPUT_FIELDS = frozenset(
    {
        "global_topology",
        "selected_topology",
        "oracle_label",
        "oracle_membership",
        "consensus_success_probability",
        "latency",
        "energy",
        "reward_surrogate",
        "future_outcome",
        "edge_delta_target",
        "critic_output",
    }
)

JUMPING_KNOWLEDGE_MODES = ("last", "concat", "max")

# Final directed-score layer init scale: keep initial logits ~0 (edge probability ~0.5,
# near-maximal entropy) so K rounds of LayerNorm do not drive the head to saturate and
# collapse at init -- the exact failure mode that pinned v3 tau at 0. See
# local_gnn_edge_scorer.py:V3_OUTPUT_HEAD_INIT_GAIN for the original lesson.
KHOP_OUTPUT_HEAD_INIT_GAIN = 0.2


@dataclass(frozen=True, slots=True)
class LocalKHopGNNEdgeScorerConfig:
    model_id: str = LOCAL_KHOP_GNN_EDGE_SCORER_V1_MODEL_ID
    node_input_dim: int = len(ACTOR_NODE_FEATURE_FIELDS)
    edge_input_dim: int = len(ACTOR_EDGE_FEATURE_FIELDS)
    hidden_dim: int = 64
    # K = number of message-passing rounds = receptive field in hops. K=3 is the
    # production candidate; K is the primary architecture ablation axis.
    rounds: int = 3
    output_dim: int = 1
    residual: bool = True
    jumping_knowledge: str = "concat"
    # Per-scene centering (PairNorm-lite). OFF by default: it trades strict
    # cross-component locality for oversmoothing control and is only an ablation.
    node_center_norm: bool = False
    dropout_probability: float = 0.0
    output_head_init_gain: float = KHOP_OUTPUT_HEAD_INIT_GAIN

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be declared")
        if self.node_input_dim <= 0:
            raise ValueError("node_input_dim must be positive")
        if self.edge_input_dim <= 0:
            raise ValueError("edge_input_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.output_dim != 1:
            raise ValueError("output_dim must be 1")
        if self.rounds < 1:
            raise ValueError("K-hop GNN requires at least one message-passing round")
        if self.jumping_knowledge not in JUMPING_KNOWLEDGE_MODES:
            raise ValueError(f"unknown jumping_knowledge mode: {self.jumping_knowledge}")
        if not 0.0 <= self.dropout_probability < 1.0:
            raise ValueError("dropout_probability must be in [0, 1)")


class LocalKHopGNNEdgeScorer(nn.Module):
    """K-round directed message-passing edge scorer over a local candidate graph.

    ``forward_graph`` consumes a (batched) candidate graph as flat tensors:
    ``node_features [N, node_dim]``, ``edge_features [E, edge_dim]``,
    ``edge_index [E, 2]`` of ``(src, dst)`` local node indices. Edges are **directed**
    and both directions of every physical link are present, so aggregation at ``dst``
    is naturally bidirectional. Multiple scenes are batched by giving disjoint node-index
    ranges per scene (``node_batch``/``edge_batch`` keep normalisation and any optional
    centering scene-local; message passing is already scene-local because ``edge_index``
    never connects two scenes).
    """

    def __init__(self, config: LocalKHopGNNEdgeScorerConfig | None = None) -> None:
        super().__init__()
        self.config = config or LocalKHopGNNEdgeScorerConfig()
        hidden = self.config.hidden_dim
        rounds = self.config.rounds
        self.node_encoder = _mlp(self.config.node_input_dim, hidden, hidden)
        self.edge_encoder = _mlp(self.config.edge_input_dim, hidden, hidden)
        # Per-round message + update blocks (edge->node messages and node->edge updates).
        self.message_layers = nn.ModuleList(
            _mlp(hidden * 3, hidden, hidden) for _ in range(rounds)
        )
        self.node_update_layers = nn.ModuleList(
            _mlp(hidden * 2, hidden, hidden) for _ in range(rounds)
        )
        self.edge_update_layers = nn.ModuleList(
            _mlp(hidden * 2, hidden, hidden) for _ in range(rounds)
        )
        self.node_norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(rounds))
        self.edge_norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(rounds))
        self.dropout = nn.Dropout(float(self.config.dropout_probability))
        # Jumping-Knowledge: the score head reads the final edge state plus (for "concat")
        # every intermediate edge state, so it can select the hop depth per edge.
        if self.config.jumping_knowledge == "concat":
            edge_summary_dim = hidden * (rounds + 1)
        else:
            edge_summary_dim = hidden
        # Directed head: [edge_summary | h_src | h_dst]; src is the deciding ("self") node.
        self.score_head = nn.Sequential(
            nn.Linear(edge_summary_dim + hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.config.output_dim),
        )
        self._reset_parameters()

    def forward_graph(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        node_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score every directed edge. Returns logits of shape ``[E]``."""

        if node_features.ndim != 2 or node_features.shape[1] != self.config.node_input_dim:
            raise ValueError("node_features must have shape [node_count, node_input_dim]")
        if edge_features.ndim != 2 or edge_features.shape[1] != self.config.edge_input_dim:
            raise ValueError("edge_features must have shape [edge_count, edge_input_dim]")
        if edge_index.ndim != 2 or edge_index.shape[1] != 2:
            raise ValueError("edge_index must have shape [edge_count, 2]")
        if edge_index.shape[0] != edge_features.shape[0]:
            raise ValueError("edge_index rows must match edge_features rows")
        num_nodes = node_features.shape[0]
        if edge_features.shape[0] == 0:
            return edge_features.new_zeros((0,))
        if num_nodes == 0:
            raise ValueError("edge_features present but node_features is empty")

        src = edge_index[:, 0].long()
        dst = edge_index[:, 1].long()
        if int(src.max().item()) >= num_nodes or int(dst.max().item()) >= num_nodes:
            raise ValueError("edge_index references a node outside node_features")

        h_node = self.node_encoder(node_features)
        h_edge = self.edge_encoder(edge_features)
        edge_states = [h_edge]

        for layer_index in range(self.config.rounds):
            # Edge->node message: each directed edge carries src/dst/edge state to dst.
            message = self.message_layers[layer_index](
                torch.cat((h_node[src], h_node[dst], h_edge), dim=1)
            )
            aggregated = _scatter_mean(message, dst, num_nodes)
            next_node = self.node_update_layers[layer_index](
                torch.cat((h_node, aggregated), dim=1)
            )
            if self.config.residual:
                h_node = h_node + self.dropout(next_node)
            else:
                h_node = self.dropout(next_node)
            h_node = self.node_norms[layer_index](h_node)
            if self.config.node_center_norm:
                h_node = _scene_center(h_node, node_batch)
            # Node->edge update: refresh each edge state from its own message.
            next_edge = self.edge_update_layers[layer_index](
                torch.cat((h_edge, message), dim=1)
            )
            if self.config.residual:
                h_edge = h_edge + self.dropout(next_edge)
            else:
                h_edge = self.dropout(next_edge)
            h_edge = self.edge_norms[layer_index](h_edge)
            edge_states.append(h_edge)

        edge_summary = self._edge_summary(edge_states)
        logits = self.score_head(
            torch.cat((edge_summary, h_node[src], h_node[dst]), dim=1)
        )
        return logits.reshape(-1)

    def score_graph_batch(self, batch: LocalGraphTensorBatch) -> torch.Tensor:
        """Score every directed edge of a tensorized candidate graph.

        Returns logits aligned row-for-row with ``batch.records`` (directed edges), so a
        logit maps to ``(owner_agent_id, neighbor_id, directed_edge_id)`` exactly like the
        flat ``ActorEdgeTensorBatch`` path. ``edge_index`` already uses global node indices
        across the batched scenes, and message passing never crosses scenes.
        """

        return self.forward_graph(
            batch.node_features,
            batch.edge_index,
            batch.edge_features,
            node_batch=batch.node_batch,
        )

    def _edge_summary(self, edge_states: list[torch.Tensor]) -> torch.Tensor:
        mode = self.config.jumping_knowledge
        if mode == "last":
            return edge_states[-1]
        if mode == "concat":
            return torch.cat(edge_states, dim=1)
        # mode == "max"
        return torch.stack(edge_states, dim=0).max(dim=0).values

    def boundary_report(self) -> dict[str, object]:
        from .local_gnn_edge_scorer import ACTIVE_STAGE33_GNN_MODEL_ID

        return {
            "model_id": self.config.model_id,
            "actor_graph_scope": "local_k_hop_candidate_graph_message_passing",
            "message_passing_layers": self.config.rounds,
            "receptive_field_hops": self.config.rounds,
            "uses_edge_to_node_messages": True,
            "uses_node_to_edge_updates": True,
            "directed_edge_head": True,
            "permutation_invariant_aggregation": "scatter_mean_by_destination_node",
            "jumping_knowledge": self.config.jumping_knowledge,
            "node_center_norm": bool(self.config.node_center_norm),
            "residual_connections": bool(self.config.residual),
            "normalization": "layer_norm_per_message_passing_layer",
            "dropout_probability": float(self.config.dropout_probability),
            "global_topology_used": False,
            "oracle_labels_used": False,
            "critic_outputs_used": False,
            "outputs_edge_scores_only": True,
            "full_message_passing_gnn": True,
            "stage33_active_production_gnn": self.config.model_id == ACTIVE_STAGE33_GNN_MODEL_ID,
        }

    @staticmethod
    def validate_input_field_names(field_names: tuple[str, ...]) -> None:
        forbidden = sorted(set(field_names) & KHOP_GNN_FORBIDDEN_INPUT_FIELDS)
        if forbidden:
            raise ValueError(f"forbidden K-hop GNN actor input fields: {forbidden}")

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        final_layer = self.score_head[-1]
        nn.init.xavier_uniform_(final_layer.weight, gain=self.config.output_head_init_gain)
        nn.init.zeros_(final_layer.bias)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
        nn.ReLU(),
    )


def _scatter_mean(values: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Mean of ``values`` grouped by destination ``index`` over ``num_nodes`` slots.

    Nodes with no incoming edge receive a zero vector (count clamped to 1). This is the
    permutation-invariant local aggregation; it never crosses scenes because ``index``
    (an edge destination) only ever references nodes within the edge's own scene.
    """

    sums = torch.zeros((num_nodes, values.shape[1]), dtype=values.dtype, device=values.device)
    sums.index_add_(0, index, values)
    counts = torch.zeros((num_nodes, 1), dtype=values.dtype, device=values.device)
    counts.index_add_(0, index, torch.ones((values.shape[0], 1), dtype=values.dtype, device=values.device))
    return sums / counts.clamp_min(1.0)


def _scene_center(h_node: torch.Tensor, node_batch: torch.Tensor | None) -> torch.Tensor:
    """Subtract the per-scene node mean (PairNorm-lite). Scene-local via ``node_batch``."""

    if node_batch is None:
        return h_node - h_node.mean(dim=0, keepdim=True)
    num_scenes = int(node_batch.max().item()) + 1
    sums = torch.zeros((num_scenes, h_node.shape[1]), dtype=h_node.dtype, device=h_node.device)
    sums.index_add_(0, node_batch, h_node)
    counts = torch.zeros((num_scenes, 1), dtype=h_node.dtype, device=h_node.device)
    counts.index_add_(
        0, node_batch, torch.ones((h_node.shape[0], 1), dtype=h_node.dtype, device=h_node.device)
    )
    means = sums / counts.clamp_min(1.0)
    return h_node - means[node_batch]
