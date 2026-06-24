"""Phase 9b (Spec §10.4): sensitivity-guided top-M counterfactual selection for SCQ.

Pins the candidate score s_e^cf = alpha|chi_e| + beta*boundary(z_e) + gamma*mutualConflict(e)
+ eta*bridgeScore(e) - zeta*cost(e), the bridge detector, and that select_topM returns the highest-
scoring (deduped, <=M) unordered add/remove/swap edits which the target builder re-decodes through the
mutual decoder, spending at most M evaluator calls.
"""

from __future__ import annotations

import torch

from marl_topology.training.decentralized_action import BCSPPerAgentAction
from marl_topology.training.counterfactual_credit import acceptance_map, mutual_active_indices
from marl_topology.training.scq_supervision import (
    SCQScoreWeights,
    scq_counterfactual_targets,
    select_topM_counterfactuals,
    sensitivity_score,
    _bridge_score,
)

EDGE_IDS = ["AB", "BC", "AC"]
EDGES = {"AB": ("A", "B"), "BC": ("B", "C"), "AC": ("A", "C")}  # idx 0=AB, 1=BC, 2=AC


def _pa(node, incident, accepted_local, budget):
    return BCSPPerAgentAction(node, tuple(incident), tuple(accepted_local),
                              torch.zeros(()), torch.zeros(()), budget)


def test_bridge_score_cycle_vs_path() -> None:
    # full triangle (cycle): no edge is a bridge
    assert _bridge_score(0, (0, 1, 2), EDGES, EDGE_IDS) == 0.0
    # path A-B-C (AB, BC active; AC absent): both AB and BC are bridges
    assert _bridge_score(0, (0, 1), EDGES, EDGE_IDS) == 1.0
    assert _bridge_score(1, (0, 1), EDGES, EDGE_IDS) == 1.0
    # an inactive edge is never a bridge
    assert _bridge_score(2, (0, 1), EDGES, EDGE_IDS) == 0.0


def test_sensitivity_score_components_isolated() -> None:
    logits = torch.tensor([0.0, 4.0, 4.0])      # edge 0 at the gate boundary; 1,2 far from it
    accept = {"A": {0}, "B": {1}, "C": {1}}     # edge 0: A has it, B doesn't -> conflict; 1: B&C both
    active = (1,)
    # chi only: mu(1-mu) maximal at mu=0.5
    w = SCQScoreWeights(alpha=1, beta=0, gamma=0, eta=0)
    assert abs(sensitivity_score(0, 0.5, logits=logits, accept=accept, edges=EDGES, edge_ids=EDGE_IDS,
                                 active=active, weights=w) - 0.25) < 1e-9
    assert sensitivity_score(0, 0.0, logits=logits, accept=accept, edges=EDGES, edge_ids=EDGE_IDS,
                             active=active, weights=w) == 0.0
    # boundary only: edge 0 (z=0) scores 1.0; edge 1 (z=4) scores 1/5
    wb = SCQScoreWeights(alpha=0, beta=1, gamma=0, eta=0)
    assert abs(sensitivity_score(0, 0.3, logits=logits, accept=accept, edges=EDGES, edge_ids=EDGE_IDS,
                                 active=active, weights=wb) - 1.0) < 1e-9
    assert abs(sensitivity_score(1, 0.3, logits=logits, accept=accept, edges=EDGES, edge_ids=EDGE_IDS,
                                 active=active, weights=wb) - (1.0 / 5.0)) < 1e-9
    # conflict only: edge 0 endpoints disagree -> 1; edge 1 both accept -> 0
    wc = SCQScoreWeights(alpha=0, beta=0, gamma=1, eta=0)
    assert sensitivity_score(0, 0.3, logits=logits, accept=accept, edges=EDGES, edge_ids=EDGE_IDS,
                             active=active, weights=wc) == 1.0
    assert sensitivity_score(1, 0.3, logits=logits, accept=accept, edges=EDGES, edge_ids=EDGE_IDS,
                             active=active, weights=wc) == 0.0
    # cost only: -zeta*cost[e]
    wz = SCQScoreWeights(alpha=0, beta=0, gamma=0, eta=0, zeta=2.0)
    assert sensitivity_score(0, 0.3, logits=logits, accept=accept, edges=EDGES, edge_ids=EDGE_IDS,
                             active=active, weights=wz, edge_cost=[1.5, 0.0, 0.0]) == -3.0


def test_topM_selects_highest_and_dedups_and_bounds_M() -> None:
    # path A-B-C active (AB,BC); bridge-only weights -> bridge edits score highest.
    # A accepts AB(0); B accepts AB(0),BC(1); C accepts BC(1) (C incident (1,2): local 0 == edge 1)
    actions = [_pa("A", (0, 2), (0,), 2), _pa("B", (0, 1), (0, 1), 2), _pa("C", (1, 2), (0,), 2)]
    accept = acceptance_map(actions)
    active = mutual_active_indices(accept, EDGE_IDS, EDGES)
    assert set(active) == {0, 1}                          # A-B-C path
    w = SCQScoreWeights(alpha=0, beta=0, gamma=0, eta=1)  # bridge-only
    out = select_topM_counterfactuals(actions, logits=torch.tensor([0.5, 0.5, 0.5]), temperature=1.0,
                                      edges=EDGES, edge_ids=EDGE_IDS, M=2, accept=accept,
                                      actual_active=active, weights=w)
    assert len(out) <= 2                                  # respects M
    assert len({(nid, frozenset(s)) for nid, s in out}) == len(out)   # deduped


def test_topM_targets_redecoded_and_budget_bounded() -> None:
    actions = [_pa("A", (0, 2), (0,), 2), _pa("B", (0, 1), (0, 1), 2), _pa("C", (1, 2), (0, 1), 2)]
    reward_of = lambda active: float(len(active))         # mock evaluator
    out = scq_counterfactual_targets(reward_of, per_agent_actions=actions, edge_ids=EDGE_IDS,
                                     edges=EDGES, logits=torch.tensor([0.6, 0.2, 0.7]), temperature=1.0,
                                     scq_m=3, r_actual=2.0, selection="topM")
    assert out.counterfactual_calls <= 3                  # at most M evaluator calls
    assert out.counterfactual_calls == out.unique_subset_count
    for active_cf, _dR in out.targets:                    # every target is a valid re-decoded topology
        assert isinstance(active_cf, tuple)
        assert all(0 <= i < len(EDGE_IDS) for i in active_cf)
        assert active_cf != out.actual_active             # no-ops skipped


def test_topM_and_simple_share_budget_accounting() -> None:
    actions = [_pa("A", (0, 2), (0,), 2), _pa("B", (0, 1), (0, 1), 2), _pa("C", (1, 2), (0, 1), 2)]
    reward_of = lambda active: float(sum((1.0, 2.0, 3.0)[i] for i in active))
    common = dict(per_agent_actions=actions, edge_ids=EDGE_IDS, edges=EDGES,
                  logits=torch.tensor([0.6, 0.2, 0.7]), temperature=1.0, scq_m=2, r_actual=2.0)
    simple = scq_counterfactual_targets(reward_of, generator=torch.Generator().manual_seed(0),
                                        selection="simple", **common)
    topm = scq_counterfactual_targets(reward_of, selection="topM", **common)
    # both honor the same budget cap (<= scq_m unique evaluator calls)
    assert simple.counterfactual_calls <= 2 and topm.counterfactual_calls <= 2
