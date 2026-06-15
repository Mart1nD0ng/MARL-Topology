"""Stage 31 Phase D: budget-aware sampler eliminates tx_budget projection friction."""

from __future__ import annotations

from collections import Counter

import torch

from marl_topology.policies.physical_link_assembler import (
    PhysicalLinkConflictAwareAssembler,
    PhysicalLinkScore,
)
from marl_topology.policies.topology_assembler import (
    AssemblerConfig,
    CandidateEdgeConstraint,
)
from marl_topology.training.policy_gradient.samplers import (
    STAGE31_ACTIVE_POLICY_GRADIENT_SAMPLER_ID,
    BudgetAwareSequentialProposalSampler,
    PhysicalPlackettLuceTopKSampler,
    ProposalSamplerConfig,
    get_stage31_active_sampler,
)

EDGES = ("n0--n1", "n0--n2", "n0--n3", "n0--n4", "n1--n2", "n3--n4")
# Logits strongly favour the four n0-incident edges, which all share endpoint n0
# and therefore collide under an endpoint budget of 1 (a worst case).
LOGITS = torch.tensor([5.0, 4.8, 4.6, 4.4, 0.1, 0.1])
MASK = torch.ones(6, dtype=torch.bool)


def _cfg(top_k: int = 4, budget: int = 1) -> ProposalSamplerConfig:
    return ProposalSamplerConfig(physical_edge_ids=EDGES, top_k=top_k, endpoint_budget=budget)


def _endpoint_usage(edges) -> Counter:
    usage: Counter = Counter()
    for edge in edges:
        a, b = edge.split("--")
        usage[a] += 1
        usage[b] += 1
    return usage


def test_budget_aware_proposals_never_exceed_endpoint_budget() -> None:
    sampler = BudgetAwareSequentialProposalSampler()
    cfg = _cfg(budget=1)
    rng = torch.Generator().manual_seed(0)
    for _ in range(200):
        sample = sampler.sample(LOGITS, MASK, cfg, rng)
        usage = _endpoint_usage(sample.proposed_physical_edges)
        assert all(count <= cfg.endpoint_budget for count in usage.values())


def test_top_k_sampler_does_violate_budget_under_stress() -> None:
    # Contrast: the active top-k sampler proposes colliding edges (the friction).
    sampler = PhysicalPlackettLuceTopKSampler()
    rng = torch.Generator().manual_seed(0)
    violations = 0
    for _ in range(50):
        sample = sampler.sample(LOGITS, MASK, _cfg(), rng)
        usage = _endpoint_usage(sample.proposed_physical_edges)
        if any(count > 1 for count in usage.values()):
            violations += 1
    assert violations > 0


def _assembler_rejections(proposed_edges) -> dict:
    scores = [
        PhysicalLinkScore(
            physical_edge_id=edge,
            endpoint_directed_edge_ids=(f"{edge.split('--')[0]}->{edge.split('--')[1]}",),
            score=1.0,
            probability=None,
        )
        for edge in proposed_edges
    ]
    constraints = [
        CandidateEdgeConstraint(
            edge_id=edge,
            tx_id=edge.split("--")[0],
            rx_id=edge.split("--")[1],
            edge_type="physical_link",
            role_allowed=True,
            channel_slot=None,
            conflict_group=f"physical_edge:{edge}",
        )
        for edge in proposed_edges
    ]
    assembler = PhysicalLinkConflictAwareAssembler(
        AssemblerConfig(
            assembler_id="stage31_test_assembler",
            mode="physical_link_conflict_aware_greedy",
            deterministic=True,
            tx_capacity=1.0,
            rx_capacity=1.0,
        )
    )
    assembly = assembler.assemble(scores, constraints)
    return dict(assembly.diagnostics["rejection_reason_counts"])


def test_budget_aware_proposal_has_zero_tx_budget_rejections() -> None:
    sampler = BudgetAwareSequentialProposalSampler()
    rng = torch.Generator().manual_seed(3)
    sample = sampler.sample(LOGITS, MASK, _cfg(budget=1), rng)
    rejections = _assembler_rejections(sample.proposed_physical_edges)
    assert rejections.get("tx_budget_exceeded", 0) == 0


def test_colliding_proposal_is_rejected_by_assembler() -> None:
    # Sanity: the assembler really does reject colliding proposals (so the
    # budget-aware sampler's zero-rejection result is meaningful, not vacuous).
    rejections = _assembler_rejections(("n0--n1", "n0--n2", "n0--n3", "n0--n4"))
    assert rejections.get("tx_budget_exceeded", 0) >= 3


def test_logprob_and_entropy_replay_is_consistent() -> None:
    sampler = BudgetAwareSequentialProposalSampler()
    rng = torch.Generator().manual_seed(7)
    sample = sampler.sample(LOGITS, MASK, _cfg(), rng)
    lp = sampler.logprob_of(LOGITS, MASK, _cfg(), sample.raw_sample_data)
    ent = sampler.entropy_of(LOGITS, MASK, _cfg(), sample.raw_sample_data)
    assert torch.allclose(lp, sample.logprob, atol=1e-6)
    assert torch.allclose(ent, sample.entropy, atol=1e-6)
    assert torch.isfinite(sample.logprob).all()
    assert torch.isfinite(sample.entropy).all()


def test_sampler_is_deterministic_given_seed() -> None:
    sampler = BudgetAwareSequentialProposalSampler()
    a = sampler.sample(LOGITS, MASK, _cfg(), torch.Generator().manual_seed(11))
    b = sampler.sample(LOGITS, MASK, _cfg(), torch.Generator().manual_seed(11))
    assert a.proposed_physical_edges == b.proposed_physical_edges


def test_stage31_active_sampler_registry() -> None:
    sampler = get_stage31_active_sampler()
    assert sampler.sampler_id == STAGE31_ACTIVE_POLICY_GRADIENT_SAMPLER_ID
