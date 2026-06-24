"""Phase 8b vs R7 credit-RESOLUTION mechanism property (Spec §13 counterfactual rank correlation).

The structural justification for COMA per-agent credit over the shared scene advantage: the shared
advantage A_s = r - V is ONE scalar handed to every agent in a scene, so its within-scene variance is
0 -- it cannot represent that agents differ in marginal contribution. The COMA credit A_i is per-agent,
so it carries non-zero within-scene variance whenever agents genuinely differ. This is a mechanism
property (NOT a corrected-env-math headline -- that is gated on the dataset rebuild).
"""

from __future__ import annotations

import torch

from marl_topology.training.counterfactual_credit import (
    per_agent_counterfactual_credit,
    within_scene_credit_variance,
)
from marl_topology.training.decentralized_action import BCSPPerAgentAction

EDGE_IDS = ["AB", "BC", "AC"]
EDGES = {"AB": ("A", "B"), "BC": ("B", "C"), "AC": ("A", "C")}


def _pa(node, incident, accepted_local, budget):
    return BCSPPerAgentAction(node, tuple(incident), tuple(accepted_local),
                              torch.zeros(()), torch.zeros(()), budget)


def test_credit_resolution_shared_is_zero_coma_is_positive() -> None:
    # asymmetric edge weights -> the three agents have genuinely different marginal contributions
    w = {0: 1.0, 1: 5.0, 2: 0.2}   # AB, BC, AC

    def q_of(active):              # a deterministic Q over the active set (the diagnostic uses the evaluator)
        return sum(w[j] for j in active)

    pa = [_pa("A", (0, 2), (0, 1), 2), _pa("B", (0, 1), (0, 1), 2), _pa("C", (1, 2), (0, 1), 2)]
    cf = per_agent_counterfactual_credit(
        q_of, per_agent_actions=pa, edge_ids=EDGE_IDS, edges=EDGES,
        logits=torch.tensor([0.4, 0.1, -0.2]), temperature=1.0, k_cf=4000,
        generator=torch.Generator().manual_seed(11))

    coma_var = within_scene_credit_variance(cf.advantages)
    # R7 shared scene advantage: ONE scalar (r - V) broadcast to every agent -> zero within-scene variance
    shared = {node: cf.q_actual - 0.5 for node in cf.advantages}
    assert within_scene_credit_variance(shared) == 0.0
    # COMA resolves per-agent credit that the shared advantage structurally cannot represent
    assert coma_var > 1e-4
    assert len({round(v, 6) for v in cf.advantages.values()}) > 1   # the A_i are genuinely distinct


def test_within_scene_variance_degenerate_cases() -> None:
    assert within_scene_credit_variance({}) == 0.0
    assert within_scene_credit_variance({"A": 0.3}) == 0.0          # a single agent -> no resolution
    assert within_scene_credit_variance({"A": 2.0, "B": 2.0}) == 0.0  # identical -> zero
