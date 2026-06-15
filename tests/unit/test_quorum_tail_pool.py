"""Contract tests for the differentiable quorum-tail readout and the opt-in critic flag.

Verifies: (1) soft_quorum_tail matches the evaluator's heterogeneous_quorum_tail on hard
inputs; (2) it is differentiable; (3) masking ignores padded nodes; (4) the critic default
pooling is byte-identical to before the flag existed (same state-dict keys, same forward);
(5) the quorum_tail critic runs, adds exactly the gate head, and produces valid outputs.
"""

from random import Random

import pytest
import torch

from marl_topology.models import CentralizedMessagePassingGraphCritic
from marl_topology.models.centralized_message_passing_graph_critic import (
    CentralizedMessagePassingGraphCriticConfig,
    GraphCriticBatch,
)
from marl_topology.models.quorum_tail_pool import (
    QUORUM_TAIL_FEATURE_DIM,
    operating_quorum,
    soft_quorum_tail,
)
from marl_topology.protocol.quorum_tail import heterogeneous_quorum_tail


def _batch(seed: int, batch_size: int = 3, nodes: int = 6, edges: int = 9):
    rng = torch.Generator().manual_seed(seed)
    node_features = torch.rand(batch_size, nodes, 8, generator=rng)
    edge_features = torch.rand(batch_size, edges, 8, generator=rng)
    edge_index = torch.randint(0, nodes, (batch_size, edges, 2), generator=rng)
    node_mask = torch.ones(batch_size, nodes)
    edge_mask = torch.ones(batch_size, edges)
    return GraphCriticBatch(node_features, edge_features, edge_index, node_mask, edge_mask)


def test_soft_quorum_tail_matches_evaluator_on_hard_inputs() -> None:
    rng = Random(0)
    for _ in range(20):
        n = rng.randint(1, 12)
        probs = [rng.random() for _ in range(n)]
        quorum = rng.randint(0, n + 1)
        gates = torch.tensor([probs])
        mask = torch.ones(1, n)
        soft = float(soft_quorum_tail(gates, mask, quorum)[0])
        hard = heterogeneous_quorum_tail(probs, quorum)
        assert abs(soft - hard) < 1e-6, f"n={n} k={quorum}: {soft} vs {hard}"


def test_soft_quorum_tail_is_differentiable() -> None:
    gates = torch.rand(2, 5, requires_grad=True)
    mask = torch.ones(2, 5)
    out = soft_quorum_tail(gates, mask, 3).sum()
    out.backward()
    assert gates.grad is not None
    assert torch.isfinite(gates.grad).all()


def test_masking_ignores_padded_nodes() -> None:
    # 3 real nodes + 2 padded: tail must equal the 3-node tail.
    probs = [0.6, 0.7, 0.8]
    gates = torch.tensor([probs + [0.9, 0.9]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
    masked = float(soft_quorum_tail(gates, mask, 2)[0])
    reference = heterogeneous_quorum_tail(probs, 2)
    assert abs(masked - reference) < 1e-6


def test_operating_quorum_matches_evaluator_and_textbook() -> None:
    n_eff = torch.tensor([4.0, 7.0, 8.0, 12.0, 16.0])
    # evaluator (fault_cap=1, the project default): commit quorum 3 for every N >= 4
    assert torch.equal(operating_quorum(n_eff), torch.tensor([3, 3, 3, 3, 3]))
    # textbook scaling 2*floor((N-1)/3)+1
    assert torch.equal(operating_quorum(n_eff, fault_cap=None), torch.tensor([3, 5, 5, 7, 11]))


def test_default_pooling_is_mean_and_byte_identical() -> None:
    assert CentralizedMessagePassingGraphCriticConfig().pooling == "mean"
    torch.manual_seed(123)
    default_critic = CentralizedMessagePassingGraphCritic()
    torch.manual_seed(123)
    explicit_mean = CentralizedMessagePassingGraphCritic(
        CentralizedMessagePassingGraphCriticConfig(pooling="mean")
    )
    # same seed + untouched mean path => identical parameters and forward.
    assert set(default_critic.state_dict()) == set(explicit_mean.state_dict())
    assert "quorum_tail_readout.gate_head.weight" not in default_critic.state_dict()
    batch = _batch(7)
    default_critic.eval()
    explicit_mean.eval()
    with torch.no_grad():
        a = default_critic(batch)
        b = explicit_mean(batch)
    assert torch.equal(a.feasibility_logit, b.feasibility_logit)
    assert torch.equal(a.consensus_proxy, b.consensus_proxy)


def test_quorum_tail_critic_runs_and_adds_only_gate_head() -> None:
    mean_keys = set(CentralizedMessagePassingGraphCritic().state_dict())
    critic = CentralizedMessagePassingGraphCritic(
        CentralizedMessagePassingGraphCriticConfig(pooling="quorum_tail")
    )
    keys = set(critic.state_dict())
    added = keys - mean_keys
    assert added == {"quorum_tail_readout.gate_head.weight", "quorum_tail_readout.gate_head.bias"}
    # graph_head input Linear grew by QUORUM_TAIL_FEATURE_DIM.
    assert critic.graph_head[0].in_features == CentralizedMessagePassingGraphCritic().graph_head[0].in_features + QUORUM_TAIL_FEATURE_DIM
    batch = _batch(11)
    critic.eval()
    out = critic(batch)
    out.assert_shapes(batch.batch_size)
    assert torch.isfinite(out.consensus_proxy).all()


def test_invalid_pooling_rejected() -> None:
    with pytest.raises(ValueError):
        CentralizedMessagePassingGraphCriticConfig(pooling="median")
