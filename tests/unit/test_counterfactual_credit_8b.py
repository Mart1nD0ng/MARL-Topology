"""Phase 8b (Spec §9.4-9.5): COMA per-agent counterfactual credit.

Pins the four correctness properties of the counterfactual advantage A_i^E = Q(s,S) - E[Q(s,S~_i,S_-i)]:
(1) a counterfactual on agent i changes the mutually-decoded active set ONLY on edges incident to i
(S_-i is held fixed and re-passed through the same decoder); (2) the per-agent advantages are not all
identical (genuine credit assignment, not the degenerate shared advantage); (3) the baseline is
INDEPENDENT of the actual S_i given (o_i, S_-i) -- the COMA unbiasedness condition; (4) on a small game
the Monte-Carlo baseline is an UNBIASED estimate of the exact E_{S~_i~pi_i}[Q(s,S~_i,S_-i)].
"""

from __future__ import annotations

import itertools
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from marl_topology.training.budget_conditioned_subset import subset_logp
from marl_topology.training.counterfactual_credit import (
    acceptance_map,
    counterfactual_advantages,
    mutual_active_indices,
)
from marl_topology.training.decentralized_action import (
    BCSPPerAgentAction,
    sample_decentralized_bcsp_action,
)
from marl_topology.models.centralized_graph_critic import CentralizedGraphCritic

# triangle A-B-C: edge index 0=AB, 1=BC, 2=AC
EDGE_IDS = ["AB", "BC", "AC"]
EDGES = {"AB": ("A", "B"), "BC": ("B", "C"), "AC": ("A", "C")}


def _incident_global(node):
    return {i for i, eid in enumerate(EDGE_IDS) if node in EDGES[eid]}


def _pa(node, incident, accepted_local, budget):
    return BCSPPerAgentAction(node, tuple(incident), tuple(accepted_local),
                              torch.zeros(()), torch.zeros(()), budget)


def test_counterfactual_changes_only_edges_incident_to_agent() -> None:
    # actual: A={AB,AC}, B={AB,BC}, C={BC,AC} -> all three edges mutually active
    actual = [_pa("A", (0, 2), (0, 1), 2), _pa("B", (0, 1), (0, 1), 2), _pa("C", (1, 2), (0, 1), 2)]
    accept = acceptance_map(actual)
    active = set(mutual_active_indices(accept, EDGE_IDS, EDGES))
    assert active == {0, 1, 2}
    # counterfactual on A: A now accepts only AB (drops AC). S_-i (B,C) fixed.
    accept_cf = dict(accept); accept_cf["A"] = {0}
    active_cf = set(mutual_active_indices(accept_cf, EDGE_IDS, EDGES))
    changed = active ^ active_cf
    assert changed <= _incident_global("A")            # only edges incident to A may change
    assert 1 not in changed                            # BC (not incident to A) is untouched


def test_other_agents_acceptance_not_mutated_by_counterfactual() -> None:
    torch.manual_seed(0)
    critic = CentralizedGraphCritic(6, 4, hidden=16, rounds=2, critic_sees_action=True)
    actual = [_pa("A", (0, 2), (0,), 2), _pa("B", (0, 1), (0, 1), 2), _pa("C", (1, 2), (0, 1), 2)]
    accept_before = acceptance_map(actual)
    cf = counterfactual_advantages(
        critic, node_features=torch.randn(3, 6), edge_features=torch.randn(3, 4),
        edge_index=torch.tensor([[0, 1], [1, 2], [0, 2]]), edge_ids=EDGE_IDS, edges=EDGES,
        per_agent_actions=actual, logits=torch.randn(3), temperature=1.0, k_cf=4,
        node_mean=torch.zeros(6), node_std=torch.ones(6), edge_mean=torch.zeros(4), edge_std=torch.ones(4),
        generator=torch.Generator().manual_seed(1))
    # the call must NOT mutate the caller's per-agent actions / acceptance (S_-i held fixed)
    assert acceptance_map(actual) == accept_before
    assert set(cf.advantages) == {"A", "B", "C"}


def test_per_agent_advantages_not_all_identical() -> None:
    torch.manual_seed(3)
    critic = CentralizedGraphCritic(6, 4, hidden=24, rounds=2, critic_sees_action=True)
    edges = {e: (u, v) for e, (u, v) in EDGES.items()}
    logits = torch.tensor([0.8, -0.4, 0.5])
    act = sample_decentralized_bcsp_action(logits, EDGE_IDS, edges=edges,
                                           budgets={"A": 2, "B": 2, "C": 2}, temperature=1.0)
    cf = counterfactual_advantages(
        critic, node_features=torch.randn(3, 6), edge_features=torch.randn(3, 4),
        edge_index=torch.tensor([[0, 1], [1, 2], [0, 2]]), edge_ids=EDGE_IDS, edges=edges,
        per_agent_actions=act.per_agent, logits=logits, temperature=1.0, k_cf=8,
        node_mean=torch.zeros(6), node_std=torch.ones(6), edge_mean=torch.zeros(4), edge_std=torch.ones(4),
        generator=torch.Generator().manual_seed(7))
    vals = list(cf.advantages.values())
    assert max(vals) - min(vals) > 1e-6     # genuine per-agent differentiation (not a shared scalar)


def test_baseline_is_independent_of_the_actual_subset() -> None:
    # two rollouts identical EXCEPT agent A's realized subset; same theta, same S_-i, same CF seed.
    torch.manual_seed(5)
    critic = CentralizedGraphCritic(6, 4, hidden=16, rounds=2, critic_sees_action=True)
    nf, ef, ei = torch.randn(3, 6), torch.randn(3, 4), torch.tensor([[0, 1], [1, 2], [0, 2]])
    logits = torch.tensor([0.6, 0.2, -0.3])
    common = dict(node_features=nf, edge_features=ef, edge_index=ei, edge_ids=EDGE_IDS, edges=EDGES,
                  logits=logits, temperature=1.0, k_cf=6, node_mean=torch.zeros(6), node_std=torch.ones(6),
                  edge_mean=torch.zeros(4), edge_std=torch.ones(4))
    pa_full = [_pa("A", (0, 2), (0, 1), 2), _pa("B", (0, 1), (0, 1), 2), _pa("C", (1, 2), (0, 1), 2)]
    pa_one = [_pa("A", (0, 2), (0,), 2), _pa("B", (0, 1), (0, 1), 2), _pa("C", (1, 2), (0, 1), 2)]
    cf_full = counterfactual_advantages(critic, per_agent_actions=pa_full,
                                        generator=torch.Generator().manual_seed(99), **common)
    cf_one = counterfactual_advantages(critic, per_agent_actions=pa_one,
                                       generator=torch.Generator().manual_seed(99), **common)
    # the COMA baseline depends only on (theta_A, b_A, S_-i) -> identical despite different actual S_A
    assert cf_full.baselines["A"] == cf_one.baselines["A"]
    # but the realized-action value Q(s,S) differs -> the advantages genuinely differ
    assert cf_full.q_actual != cf_one.q_actual


class _MockQ:
    """A deterministic action-conditioned Q over the active-edge one-hot (for the unbiasedness check)."""
    critic_sees_action = True

    def __init__(self, weights):
        self.w = weights

    def __call__(self, nf, ef, edge_index, node_mask, edge_mask, active_edge_onehot=None):
        oh = active_edge_onehot[0]
        q = (self.w * oh).sum() + 0.7 * oh[0] * oh[1]   # a non-linear function of the active set
        return q.reshape(1)


def test_small_game_counterfactual_baseline_is_unbiased() -> None:
    # one agent "A" with all incident edges; B and C accept everything -> active(S~_A) is decided by A.
    edge_ids = ["AB", "AC"]
    edges = {"AB": ("A", "B"), "AC": ("A", "C")}
    logits = torch.tensor([0.9, -0.2], dtype=torch.float64)
    budget = 2
    theta = logits  # T = 1
    critic = _MockQ(torch.tensor([1.3, -0.8], dtype=torch.float64))

    # exact E_{S~_A ~ pi_A}[Q(active(S~_A, S_-A))] by enumerating A's legal subsets weighted by pi_A
    local = list(range(2))
    exact = 0.0
    for r in range(budget + 1):
        for subset in itertools.combinations(local, r):
            w = math.exp(float(subset_logp(theta, subset, budget)))
            accept = {"A": {(0, 1)[j] for j in subset}, "B": {0}, "C": {1}}  # B,C accept their edge
            active = mutual_active_indices(accept, edge_ids, edges)
            oh = torch.zeros(2, dtype=torch.float64)
            if active:
                oh[list(active)] = 1.0
            q = float((critic.w * oh).sum() + 0.7 * oh[0] * oh[1])
            exact += w * q

    # Monte-Carlo baseline from the function, large K_cf -> should converge to the exact expectation
    pa = [_pa("A", (0, 1), (0,), budget), _pa("B", (0,), (0,), 1), _pa("C", (1,), (0,), 1)]
    cf = counterfactual_advantages(
        critic, node_features=torch.zeros(3, 2, dtype=torch.float64),
        edge_features=torch.zeros(2, 2, dtype=torch.float64),
        edge_index=torch.tensor([[0, 1], [0, 2]]), edge_ids=edge_ids, edges=edges,
        per_agent_actions=pa, logits=logits, temperature=1.0, k_cf=20000,
        node_mean=torch.zeros(2, dtype=torch.float64), node_std=torch.ones(2, dtype=torch.float64),
        edge_mean=torch.zeros(2, dtype=torch.float64), edge_std=torch.ones(2, dtype=torch.float64),
        generator=torch.Generator().manual_seed(2024))
    assert abs(cf.baselines["A"] - exact) < 0.02   # MC COMA baseline == exact expectation (unbiased)


@pytest.mark.slow
def test_resume_rejects_critic_architecture_mismatch(tmp_path) -> None:
    # 8b adversarial-verify follow-up: a --resume that forgets --counterfactual must fail with a CLEAR
    # message (the Q critic's edge_dim+1 encoder != the V critic's), not a cryptic shape RuntimeError.
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(root / "src")}
    script = str(root / "scripts" / "train" / "train_decentralized_rl.py")
    base = [sys.executable, script, "--smoke", "--cold-start", "--baseline", "graph-mappo",
            "--ckpt-every", "1", "--out-dir", str(tmp_path)]
    run1 = subprocess.run(base + ["--counterfactual"], capture_output=True, text=True,
                          timeout=240, env=env, cwd=str(root))
    assert run1.returncode == 0, run1.stdout[-1500:] + run1.stderr[-1500:]
    # resume WITHOUT --counterfactual -> the guard must trip (non-zero exit + the explanatory message)
    run2 = subprocess.run(base + ["--resume"], capture_output=True, text=True,
                          timeout=240, env=env, cwd=str(root))
    out = run2.stdout + run2.stderr
    assert run2.returncode != 0
    assert "critic_sees_action" in out and "--counterfactual" in out
