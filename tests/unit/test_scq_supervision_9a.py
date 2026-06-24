"""Phase 9a (Spec §10.2/§10.5): SCQ closed-form counterfactual supervision for the Q critic.

Pins: L_SCQ = mean_i [(Q(s,S)-Q(s,S~_i,S_-i)) - DeltaR_i^env]^2 is ZERO exactly when the critic's
predicted difference equals the EXACT evaluator difference; DeltaR comes from the REAL evaluator
(reward_of), not the learned Q; the loss is grad-on for the critic (and involves no actor); the extra
evaluator calls are counted honestly (budget); and on a small game L_SCQ training calibrates a real
Q critic toward the exact difference. Counterfactuals are unordered subsets re-passed through the
mutual decoder (reused, verified).
"""

from __future__ import annotations

import torch

from marl_topology.training.decentralized_action import BCSPPerAgentAction
from marl_topology.training.scq_supervision import scq_consistency_loss
from marl_topology.models.centralized_graph_critic import CentralizedGraphCritic
from marl_topology.training.graph_mappo import critic_q_value

EDGE_IDS = ["AB", "BC", "AC"]
EDGES = {"AB": ("A", "B"), "BC": ("B", "C"), "AC": ("A", "C")}


def _pa(node, incident, accepted_local, budget):
    return BCSPPerAgentAction(node, tuple(incident), tuple(accepted_local),
                              torch.zeros(()), torch.zeros(()), budget)


def _actions():
    # actual: A={AB}, B={AB,BC}, C={BC,AC} -> active {0,1} (AC inactive: A doesn't accept it)
    return [_pa("A", (0, 2), (0,), 2), _pa("B", (0, 1), (0, 1), 2), _pa("C", (1, 2), (0, 1), 2)]


def _weighted_q(weights):
    W = torch.nn.Parameter(weights.clone())

    def q_of(active):
        if not active:
            return W.sum() * 0.0
        return torch.stack([W[j] for j in active]).sum()
    return W, q_of


def test_scq_loss_zero_when_Q_matches_exact_diff() -> None:
    w = torch.tensor([1.0, 5.0, 0.2])
    W, q_of = _weighted_q(w)
    reward_of = lambda active: float(sum(float(w[j]) for j in active))  # SAME weights -> Q_diff==R_diff
    out = scq_consistency_loss(q_of, reward_of, per_agent_actions=_actions(), edge_ids=EDGE_IDS,
                               edges=EDGES, logits=torch.tensor([0.9, 0.3, 0.8]), temperature=1.0,
                               scq_m=3, generator=torch.Generator().manual_seed(3))
    assert float(out.loss) < 1e-10                 # residual (Q_diff - DeltaR) is exactly 0
    assert out.mean_abs_residual < 1e-6


def test_scq_delta_comes_from_real_evaluator_not_Q() -> None:
    # constant Q -> Q_diff == 0, so any nonzero loss must come ENTIRELY from reward_of's DeltaR.
    W, q_of = _weighted_q(torch.zeros(3))          # q_of(active)=0 for every active set
    reward_of = lambda active: float(sum((1.3, -0.8, 0.5)[j] for j in active))
    out = scq_consistency_loss(q_of, reward_of, per_agent_actions=_actions(), edge_ids=EDGE_IDS,
                               edges=EDGES, logits=torch.tensor([2.0, 0.3, 2.0]), temperature=1.0,
                               scq_m=3, generator=torch.Generator().manual_seed(1))
    assert out.counterfactual_calls >= 1           # non-vacuous: a real counterfactual fired
    assert float(out.loss) > 1e-6                   # loss = mean(DeltaR^2) > 0, driven by the evaluator


def test_scq_loss_is_grad_on_for_critic() -> None:
    W, q_of = _weighted_q(torch.tensor([0.5, 0.5, 0.5]))
    reward_of = lambda active: float(len(active))  # arbitrary evaluator
    out = scq_consistency_loss(q_of, reward_of, per_agent_actions=_actions(), edge_ids=EDGE_IDS,
                               edges=EDGES, logits=torch.tensor([2.0, 0.3, 2.0]), temperature=1.0,
                               scq_m=3, generator=torch.Generator().manual_seed(1))
    assert out.loss.requires_grad and out.loss.grad_fn is not None
    out.loss.backward()
    assert W.grad is not None and torch.any(W.grad != 0)   # SCQ trains the critic params


def test_scq_counterfactual_calls_counted_and_budget_honest() -> None:
    W, q_of = _weighted_q(torch.tensor([1.0, 2.0, 3.0]))
    reward_of = lambda active: float(sum((1.0, 2.0, 3.0)[j] for j in active))
    out = scq_consistency_loss(q_of, reward_of, per_agent_actions=_actions(), edge_ids=EDGE_IDS,
                               edges=EDGES, logits=torch.tensor([2.0, 0.3, 2.0]), temperature=1.0,
                               scq_m=2, generator=torch.Generator().manual_seed(1))
    # at most scq_m unique non-trivial counterfactuals; each unique one == exactly one evaluator call
    assert out.counterfactual_calls <= 2
    assert out.counterfactual_calls == out.unique_subset_count


def test_scq_r_actual_reused_not_recomputed() -> None:
    # passing r_actual must skip the actual-action evaluator call (it's reused from the rollout)
    calls = {"n": 0}

    def reward_of(active):
        calls["n"] += 1
        return float(len(active))
    W, q_of = _weighted_q(torch.tensor([0.5, 0.5, 0.5]))
    out = scq_consistency_loss(q_of, reward_of, per_agent_actions=_actions(), edge_ids=EDGE_IDS,
                               edges=EDGES, logits=torch.tensor([2.0, 0.3, 2.0]), temperature=1.0,
                               scq_m=3, r_actual=2.0, generator=torch.Generator().manual_seed(1))
    # reward_of called ONLY for counterfactuals, never for the actual action (r_actual supplied)
    assert calls["n"] == out.counterfactual_calls


def test_scq_small_game_calibrates_real_q_critic() -> None:
    # a real action-conditioned Q critic, trained with L_SCQ against a deterministic evaluator, should
    # drive its predicted difference toward the exact evaluator difference (loss decreases).
    torch.manual_seed(0)
    ND, ED, N = 6, 4, 3
    nf, ef = torch.randn(N, ND), torch.randn(len(EDGE_IDS), ED)
    ei = torch.tensor([[0, 1], [1, 2], [0, 2]])
    nm, em = (torch.zeros(ND), torch.ones(ND)), (torch.zeros(ED), torch.ones(ED))
    critic = CentralizedGraphCritic(ND, ED, hidden=24, rounds=2, critic_sees_action=True)
    opt = torch.optim.AdamW(critic.parameters(), lr=5e-3)
    w = (1.3, -0.8, 0.5)
    reward_of = lambda active: float(sum(w[j] for j in active))

    def q_of(active):
        oh = torch.zeros(len(EDGE_IDS))
        if active:
            oh[list(active)] = 1.0
        return critic_q_value(critic, nf, ef, ei, oh, node_mean=nm[0], node_std=nm[1],
                              edge_mean=em[0], edge_std=em[1])

    losses = []
    for step in range(120):
        out = scq_consistency_loss(q_of, reward_of, per_agent_actions=_actions(), edge_ids=EDGE_IDS,
                                   edges=EDGES, logits=torch.tensor([2.0, 0.3, 2.0]), temperature=1.0,
                                   scq_m=3, generator=torch.Generator().manual_seed(step))
        if out.loss.requires_grad:
            opt.zero_grad(); out.loss.backward(); opt.step()
        losses.append(float(out.loss))
    early = sum(losses[:20]) / 20
    late = sum(losses[-20:]) / 20
    assert late < 0.6 * early    # SCQ calibrates Q: predicted difference converges to the exact diff
