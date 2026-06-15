import torch
import pytest

from marl_topology.training.policy_gradient.samplers import (
    ENDPOINT_BUDGETED_PHYSICAL_PROPOSAL_SAMPLER_ID,
    EndpointBudgetedPhysicalProposalSampler,
    ProposalSamplerConfig,
)


def _config() -> ProposalSamplerConfig:
    return ProposalSamplerConfig(
        physical_edge_ids=("a--b", "a--c", "b--c", "c--d"),
        top_k=2,
        endpoint_budget=1,
    )


def test_stage24_endpoint_sampler_records_exact_endpoint_action_logprob() -> None:
    sampler = EndpointBudgetedPhysicalProposalSampler()
    logits = torch.tensor([2.0, 0.5, -1.0, 1.25])
    mask = torch.tensor([True, False, True, True])

    sample = sampler.sample(logits, mask, _config(), torch.Generator().manual_seed(24))
    recomputed = sampler.logprob_of(logits, mask, _config(), sample.raw_sample_data)
    entropy = sampler.entropy_of(logits, mask, _config(), sample.raw_sample_data)

    assert sample.sampler_id == ENDPOINT_BUDGETED_PHYSICAL_PROPOSAL_SAMPLER_ID
    assert torch.allclose(sample.logprob, recomputed)
    assert torch.isfinite(sample.logprob).item()
    assert torch.isfinite(entropy).item()
    assert sample.raw_sample_data["logprob_semantics"] == "exact_endpoint_proposal_logprob"
    assert sample.raw_sample_data["projected_topology_logprob_exact"] is False
    assert sample.diagnostics["endpoint_choice_logprob_exact"] is True
    assert sample.diagnostics["aggregated_physical_logprob_claimed_exact"] is False
    assert sample.diagnostics["objective_or_reward_inputs_used"] is False


def test_stage24_endpoint_sampler_budget_mask_and_aggregate_are_deterministic() -> None:
    sampler = EndpointBudgetedPhysicalProposalSampler()
    logits = torch.tensor([2.0, 0.5, -1.0, 1.25])
    mask = torch.tensor([True, False, True, True])
    config = _config()

    first = sampler.sample(logits, mask, config, torch.Generator().manual_seed(99))
    second = sampler.sample(logits, mask, config, torch.Generator().manual_seed(99))

    assert first.proposed_physical_edges == second.proposed_physical_edges
    assert "a--c" not in first.proposed_physical_edges
    endpoint_sets = first.raw_sample_data["endpoint_proposal_sets"]
    assert all(len(edges) <= config.endpoint_budget for edges in endpoint_sets.values())
    assert tuple(first.raw_sample_data["aggregated_physical_proposals"]) == first.proposed_physical_edges
    with pytest.raises(ValueError, match="forbidden sampler input fields"):
        ProposalSamplerConfig(
            physical_edge_ids=("a--b",),
            forbidden_input_fields=("consensus_success_probability",),
        )
