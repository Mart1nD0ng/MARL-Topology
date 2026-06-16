"""Tests for the decentralized per-node mutual-acceptance sampler (Task 2 decoder).

Headline guarantees:
- mutual acceptance: an edge activates iff BOTH endpoints pick each other;
- factorized joint log-prob with NO double counting: sum of per-agent log-probs equals
  the scalar joint log-prob, and the replayed log-prob equals the sampled one (PPO-exact);
- per-node radio budget respected by construction; STOP token allows an empty proposal.
"""

import math

import pytest
import torch

from marl_topology.training.policy_gradient.decentralized_sampler import (
    DECENTRALIZED_PER_NODE_MUTUAL_SAMPLER_ID,
    DecentralizedMutualSamplerConfig,
    DecentralizedPerNodeMutualSampler,
)


# Triangle a,b,c. Directed order fixed so we can address logits by index.
DIRECTED = ("a->b", "a->c", "b->a", "b->c", "c->a", "c->b")


def _config(**overrides) -> DecentralizedMutualSamplerConfig:
    overrides.setdefault("default_budget", 2)
    return DecentralizedMutualSamplerConfig(directed_edge_ids=DIRECTED, **overrides)


def _mask() -> torch.Tensor:
    return torch.ones(len(DIRECTED), dtype=torch.bool)


def test_mutual_acceptance_requires_both_endpoints_deterministic() -> None:
    # a<->b both want each other (mutual); a wants c but c rejects a; b,c reject each other.
    logits = torch.tensor([5.0, 5.0, 5.0, -5.0, -5.0, -5.0])  # a->b,a->c,b->a,b->c,c->a,c->b
    sampler = DecentralizedPerNodeMutualSampler()
    sample = sampler.sample(logits, _mask(), _config(), torch.Generator().manual_seed(1), deterministic=True)
    assert sample.proposed_physical_edges == ("a--b",)
    assert sample.sampler_id == DECENTRALIZED_PER_NODE_MUTUAL_SAMPLER_ID


def test_one_sided_preference_is_not_activated() -> None:
    # a->c very high, c->a very low: a wants c, c does not want a -> no a--c edge.
    logits = torch.tensor([-5.0, 9.0, -5.0, -5.0, -9.0, -5.0])
    sampler = DecentralizedPerNodeMutualSampler()
    sample = sampler.sample(logits, _mask(), _config(), torch.Generator().manual_seed(2), deterministic=True)
    assert "a--c" not in sample.proposed_physical_edges


def test_stop_token_allows_empty_proposal() -> None:
    logits = torch.full((len(DIRECTED),), -10.0)
    sampler = DecentralizedPerNodeMutualSampler()
    sample = sampler.sample(logits, _mask(), _config(), torch.Generator().manual_seed(3), deterministic=True)
    assert sample.proposed_physical_edges == ()


def test_logprob_factorizes_with_no_double_counting() -> None:
    torch.manual_seed(0)
    logits = torch.randn(len(DIRECTED))
    sampler = DecentralizedPerNodeMutualSampler()
    config = _config()
    mask = _mask()
    sample = sampler.sample(logits, mask, config, torch.Generator().manual_seed(7))

    # 1) Replayed joint log-prob equals sampled joint log-prob (PPO-exact).
    recomputed = sampler.logprob_of(logits, mask, config, sample.raw_sample_data)
    assert torch.allclose(sample.logprob, recomputed, atol=1e-6)

    # 2) Sum of per-agent log-probs equals the joint log-prob (no term counted twice).
    per_owner = sampler.per_owner_logprobs(logits, mask, config, sample.raw_sample_data)
    assert torch.allclose(sum(per_owner.values()), sample.logprob, atol=1e-6)

    # 3) The per-owner log-probs recorded at sample time also sum to the joint.
    recorded = sample.raw_sample_data["per_owner_logprobs"]
    assert math.isclose(sum(recorded.values()), float(sample.logprob), abs_tol=1e-5)


def test_entropy_of_matches_sample() -> None:
    torch.manual_seed(1)
    logits = torch.randn(len(DIRECTED))
    sampler = DecentralizedPerNodeMutualSampler()
    config = _config()
    mask = _mask()
    sample = sampler.sample(logits, mask, config, torch.Generator().manual_seed(11))
    recomputed = sampler.entropy_of(logits, mask, config, sample.raw_sample_data)
    assert torch.allclose(sample.entropy, recomputed, atol=1e-6)


def test_reproducible_with_seed() -> None:
    torch.manual_seed(2)
    logits = torch.randn(len(DIRECTED))
    sampler = DecentralizedPerNodeMutualSampler()
    config = _config()
    mask = _mask()
    first = sampler.sample(logits, mask, config, torch.Generator().manual_seed(99))
    second = sampler.sample(logits, mask, config, torch.Generator().manual_seed(99))
    assert first.proposed_physical_edges == second.proposed_physical_edges
    assert torch.allclose(first.logprob, second.logprob)


def test_budget_respected_by_construction() -> None:
    logits = torch.full((len(DIRECTED),), 8.0)  # everyone wants everyone
    sampler = DecentralizedPerNodeMutualSampler()
    config = _config(default_budget=1)
    sample = sampler.sample(logits, _mask(), config, torch.Generator().manual_seed(5), deterministic=True)
    picks = sample.raw_sample_data["ordered_indices_by_owner"]
    for owner, owner_picks in picks.items():
        assert len(owner_picks) <= 1, f"node {owner} exceeded its radio budget"


def test_per_node_budget_override() -> None:
    logits = torch.full((len(DIRECTED),), 8.0)
    sampler = DecentralizedPerNodeMutualSampler()
    # a gets budget 2, b and c budget 1.
    config = _config(default_budget=1, node_budgets=(("a", 2),))
    sample = sampler.sample(logits, _mask(), config, torch.Generator().manual_seed(5), deterministic=True)
    picks = sample.raw_sample_data["ordered_indices_by_owner"]
    assert len(picks.get("a", ())) == 2
    assert len(picks.get("b", ())) <= 1
    assert len(picks.get("c", ())) <= 1


def test_config_rejects_forbidden_fields() -> None:
    with pytest.raises(ValueError, match="forbidden sampler input fields"):
        DecentralizedMutualSamplerConfig(
            directed_edge_ids=DIRECTED,
            forbidden_input_fields=("consensus_success_probability",),
        )


def test_config_rejects_bad_directed_ids() -> None:
    with pytest.raises(ValueError):
        DecentralizedMutualSamplerConfig(directed_edge_ids=("a--b",))  # not directed form
    with pytest.raises(ValueError):
        DecentralizedMutualSamplerConfig(directed_edge_ids=())


def test_batch_sampling() -> None:
    logits = torch.randn(2, len(DIRECTED))
    mask = torch.ones(2, len(DIRECTED), dtype=torch.bool)
    sampler = DecentralizedPerNodeMutualSampler()
    samples = sampler.sample_batch(logits, mask, _config(), torch.Generator().manual_seed(1))
    assert len(samples) == 2
    assert all(torch.isfinite(s.logprob).item() for s in samples)
