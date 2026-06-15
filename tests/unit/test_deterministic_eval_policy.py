"""Deterministic (mode/argmax) action selection for EVALUATION.

The active Plackett-Luce top-k sampler gets a `deterministic` flag: instead of sampling,
it greedily takes the highest-logit edge at each step -> the policy MODE, i.e. the action
a deployed controller would take. This is eval-only (training stays stochastic), so the
PPO ratio is untouched. Diagnostics showed the mode reaches the feasibility ceiling while
stochastic sampling under-reports it ~3x.
"""

import torch

from marl_topology.training.policy_gradient.samplers import (
    PhysicalPlackettLuceTopKSampler,
    ProposalSamplerConfig,
)
from marl_topology.training.production_mappo_adapter import Stage33GNNStabilityConfig


def _cfg(edge_ids, top_k):
    return ProposalSamplerConfig(physical_edge_ids=edge_ids, top_k=top_k, endpoint_budget=1)


def test_deterministic_selects_top_k_by_logit() -> None:
    sampler = PhysicalPlackettLuceTopKSampler()
    logits = torch.tensor([0.1, 5.0, 0.2, 4.0, 0.3])
    mask = torch.ones(5, dtype=torch.bool)
    cfg = _cfg(("a--n0", "a--n1", "a--n2", "a--n3", "a--n4"), top_k=2)
    sample = sampler.sample(logits, mask, cfg, torch.Generator().manual_seed(0), deterministic=True)
    # the two highest-logit edges are a--n1 (5.0) and a--n3 (4.0).
    assert set(sample.proposed_physical_edges) == {"a--n1", "a--n3"}


def test_deterministic_is_rng_invariant() -> None:
    sampler = PhysicalPlackettLuceTopKSampler()
    logits = torch.tensor([0.1, 5.0, 0.2, 4.0, 0.3])
    mask = torch.ones(5, dtype=torch.bool)
    cfg = _cfg(("a--n0", "a--n1", "a--n2", "a--n3", "a--n4"), top_k=3)
    a = sampler.sample(logits, mask, cfg, torch.Generator().manual_seed(1), deterministic=True)
    b = sampler.sample(logits, mask, cfg, torch.Generator().manual_seed(999), deterministic=True)
    # the mode does not depend on the rng.
    assert a.proposed_physical_edges == b.proposed_physical_edges


def test_stochastic_is_still_the_default() -> None:
    sampler = PhysicalPlackettLuceTopKSampler()
    logits = torch.tensor([0.1, 5.0, 0.2, 4.0, 0.3])
    mask = torch.ones(5, dtype=torch.bool)
    cfg = _cfg(("a--n0", "a--n1", "a--n2", "a--n3", "a--n4"), top_k=2)
    # default (no flag) keeps sampling: a valid top_k proposal, logprob/entropy present.
    sample = sampler.sample(logits, mask, cfg, torch.Generator().manual_seed(3))
    assert len(sample.proposed_physical_edges) == 2
    assert sample.logprob is not None and sample.entropy is not None


def test_deterministic_eval_flag_defaults_off_and_serializes() -> None:
    assert Stage33GNNStabilityConfig().deterministic_eval is False
    payload = Stage33GNNStabilityConfig(deterministic_eval=True).to_payload()
    assert payload["deterministic_eval"] is True
    # the frozen reward contract is untouched by the eval-protocol flag.
    assert payload["reward_weights_tuned"] is False
