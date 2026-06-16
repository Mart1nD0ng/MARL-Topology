"""Authoritative production-trunk designation.

After the decentralized-MARL consolidation the production model is the decentralized CTDE
flow, NOT the Stage-33 ego-graph + global-Plackett-Luce path. This module is the single
source of truth a reviewer can read to know which components are load-bearing production:

* PRODUCTION_TRUNK_ACTOR_MODEL_ID  -- the deployment actor (K-hop local message-passing GNN
  edge scorer): genuinely multi-hop yet Dec-POMDP-local (receptive field = K hops via local
  neighbour signalling, no global-state shortcut).
* PRODUCTION_TRUNK_SAMPLER_ID      -- the decentralized per-node mutual-acceptance decoder
  (each node decides over its own incident edges within its radio budget; an edge activates
  iff both endpoints accept). True end-to-end decentralized decode.
* PRODUCTION_TRUNK_FLOW_ID         -- the decentralized CTDE multi-agent policy-gradient flow
  (per-agent clipped objective from factorized per-owner log-probs + a shared training-only
  centralized critic; genuinely sequential via the per-node energy battery, not a bandit).

The former Stage-33 designation (``ACTIVE_STAGE33_GNN_MODEL_ID`` = v3 residual-norm ego-graph
scorer; ``ACTIVE_POLICY_GRADIENT_SAMPLER_ID`` = global Plackett-Luce top-k) is retained only
as the DIAGNOSTIC BASELINE / ablation the production trunk is measured against. It is not the
production model. See ``docs/TRUNK_MAP.md``.
"""

from __future__ import annotations

from marl_topology.models.local_gnn_edge_scorer import (
    ACTIVE_STAGE33_GNN_MODEL_ID,
    LOCAL_GNN_V3_RESIDUAL_NORM_MODEL_ID,
)
from marl_topology.models.local_khop_gnn_edge_scorer import (
    LOCAL_KHOP_GNN_EDGE_SCORER_V1_MODEL_ID,
)
from marl_topology.training.decentralized_marl import DECENTRALIZED_CTDE_FLOW_ID
from marl_topology.training.policy_gradient.decentralized_sampler import (
    DECENTRALIZED_PER_NODE_MUTUAL_SAMPLER_ID,
)
from marl_topology.training.policy_gradient.samplers import (
    ACTIVE_POLICY_GRADIENT_SAMPLER_ID,
)


PRODUCTION_TRUNK_ACTOR_MODEL_ID = LOCAL_KHOP_GNN_EDGE_SCORER_V1_MODEL_ID
PRODUCTION_TRUNK_SAMPLER_ID = DECENTRALIZED_PER_NODE_MUTUAL_SAMPLER_ID
PRODUCTION_TRUNK_FLOW_ID = DECENTRALIZED_CTDE_FLOW_ID

# The legacy ego-graph + global-decode path, kept as the diagnostic baseline.
DIAGNOSTIC_BASELINE_ACTOR_MODEL_ID = LOCAL_GNN_V3_RESIDUAL_NORM_MODEL_ID
DIAGNOSTIC_BASELINE_SAMPLER_ID = ACTIVE_POLICY_GRADIENT_SAMPLER_ID


def production_trunk_designation() -> dict[str, object]:
    """Authoritative mapping of production trunk vs. retained diagnostic baseline."""

    return {
        "production_trunk_flow_id": PRODUCTION_TRUNK_FLOW_ID,
        "production_trunk_actor_model_id": PRODUCTION_TRUNK_ACTOR_MODEL_ID,
        "production_trunk_sampler_id": PRODUCTION_TRUNK_SAMPLER_ID,
        "production_decode_is_decentralized": True,
        "production_execution_is_decentralized": True,
        "production_is_sequential_mdp": True,
        "production_is_bandit": False,
        "diagnostic_baseline_actor_model_id": DIAGNOSTIC_BASELINE_ACTOR_MODEL_ID,
        "diagnostic_baseline_sampler_id": DIAGNOSTIC_BASELINE_SAMPLER_ID,
        "legacy_stage33_active_gnn_is_diagnostic_only": (
            ACTIVE_STAGE33_GNN_MODEL_ID == DIAGNOSTIC_BASELINE_ACTOR_MODEL_ID
        ),
    }
