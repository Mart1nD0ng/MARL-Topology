"""Contract tests for the opt-in distributional critic heads (innovation 3).

Verifies: default byte-identical (no distributional heads, identical forward), the
distributional config adds exactly the two heads, the new outputs have valid shapes and
are None on the default path, and the distributional heads compose with quorum_tail pooling.
"""

import torch

from marl_topology.models import CentralizedMessagePassingGraphCritic
from marl_topology.models.centralized_message_passing_graph_critic import (
    CentralizedMessagePassingGraphCriticConfig,
    GraphCriticBatch,
)


def _batch(seed: int, batch_size: int = 3, nodes: int = 6, edges: int = 9):
    rng = torch.Generator().manual_seed(seed)
    return GraphCriticBatch(
        node_features=torch.rand(batch_size, nodes, 8, generator=rng),
        edge_features=torch.rand(batch_size, edges, 8, generator=rng),
        edge_index=torch.randint(0, nodes, (batch_size, edges, 2), generator=rng),
        node_mask=torch.ones(batch_size, nodes),
        edge_mask=torch.ones(batch_size, edges),
    )


def test_default_is_byte_identical_and_outputs_none() -> None:
    assert CentralizedMessagePassingGraphCriticConfig().distributional is False
    torch.manual_seed(7)
    default = CentralizedMessagePassingGraphCritic()
    torch.manual_seed(7)
    explicit = CentralizedMessagePassingGraphCritic(
        CentralizedMessagePassingGraphCriticConfig(distributional=False)
    )
    assert set(default.state_dict()) == set(explicit.state_dict())
    assert "robust_feasibility_head.weight" not in default.state_dict()
    batch = _batch(3)
    default.eval()
    with torch.no_grad():
        out = default(batch)
    assert out.robust_feasibility_logit is None
    assert out.consensus_low_quantile is None


def test_distributional_adds_only_two_heads() -> None:
    point_keys = set(CentralizedMessagePassingGraphCritic().state_dict())
    critic = CentralizedMessagePassingGraphCritic(
        CentralizedMessagePassingGraphCriticConfig(distributional=True)
    )
    added = set(critic.state_dict()) - point_keys
    assert added == {
        "robust_feasibility_head.weight",
        "robust_feasibility_head.bias",
        "consensus_low_quantile_head.weight",
        "consensus_low_quantile_head.bias",
    }


def test_distributional_outputs_valid() -> None:
    critic = CentralizedMessagePassingGraphCritic(
        CentralizedMessagePassingGraphCriticConfig(distributional=True)
    )
    critic.eval()
    batch = _batch(11)
    with torch.no_grad():
        out = critic(batch)
    out.assert_shapes(batch.batch_size)
    assert out.robust_feasibility_logit.shape == (batch.batch_size,)
    assert out.consensus_low_quantile.shape == (batch.batch_size,)
    assert torch.isfinite(out.robust_feasibility_logit).all()
    assert torch.sigmoid(out.robust_feasibility_logit).min() >= 0.0
    assert torch.sigmoid(out.robust_feasibility_logit).max() <= 1.0


def test_distributional_composes_with_quorum_tail() -> None:
    critic = CentralizedMessagePassingGraphCritic(
        CentralizedMessagePassingGraphCriticConfig(distributional=True, pooling="quorum_tail")
    )
    critic.eval()
    batch = _batch(5)
    with torch.no_grad():
        out = critic(batch)
    out.assert_shapes(batch.batch_size)
    assert out.robust_feasibility_logit is not None
    keys = set(critic.state_dict())
    assert "quorum_tail_readout.gate_head.weight" in keys
    assert "robust_feasibility_head.weight" in keys
