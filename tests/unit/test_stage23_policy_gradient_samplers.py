import math

import torch
import pytest

from marl_topology.training.policy_gradient.samplers import (
    ACTIVE_POLICY_GRADIENT_SAMPLER_ID,
    ENDPOINT_BUDGETED_PHYSICAL_PROPOSAL_SAMPLER_ID,
    PHYSICAL_BERNOULLI_PROPOSAL_SAMPLER_ID,
    PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID,
    ProposalSamplerConfig,
    build_active_policy_gradient_sampler_registry,
    build_stage23_trial_sampler_registry,
)


def _config() -> ProposalSamplerConfig:
    return ProposalSamplerConfig(
        physical_edge_ids=(
            "a--b",
            "a--c",
            "b--c",
            "c--d",
        ),
        top_k=2,
        endpoint_budget=1,
    )


def test_stage23_trial_samplers_have_finite_reproducible_proposal_math() -> None:
    logits = torch.tensor([2.0, 0.5, -1.0, 1.25])
    mask = torch.tensor([True, False, True, True])
    config = _config()

    for sampler_id, sampler in build_stage23_trial_sampler_registry().items():
        first = sampler.sample(
            logits,
            mask,
            config,
            torch.Generator().manual_seed(23),
        )
        second = sampler.sample(
            logits,
            mask,
            config,
            torch.Generator().manual_seed(23),
        )

        assert first.sampler_id == sampler_id
        assert first.proposed_physical_edges == second.proposed_physical_edges
        assert torch.isfinite(first.logprob).item()
        assert torch.isfinite(first.entropy).item()
        assert math.isfinite(first.to_payload()["logprob"])
        assert set(first.proposed_physical_edges).isdisjoint({"a--c"})
        recomputed = sampler.logprob_of(
            logits,
            mask,
            config,
            first.raw_sample_data,
        )
        assert torch.allclose(first.logprob, recomputed)


def test_stage23_samplers_support_repeated_batch_calls() -> None:
    logits = torch.tensor(
        [
            [2.0, 0.5, -1.0, 1.25],
            [1.0, -0.25, 0.75, 0.0],
        ]
    )
    mask = torch.tensor(
        [
            [True, False, True, True],
            [True, True, False, True],
        ]
    )
    config = _config()

    for sampler in build_stage23_trial_sampler_registry().values():
        samples = sampler.sample_batch(
            logits,
            mask,
            config,
            torch.Generator().manual_seed(123),
        )
        assert len(samples) == 2
        assert all(torch.isfinite(sample.logprob).item() for sample in samples)


def test_stage23_sampler_config_rejects_objective_or_reward_leakage() -> None:
    with pytest.raises(ValueError, match="forbidden sampler input fields"):
        ProposalSamplerConfig(
            physical_edge_ids=("a--b",),
            forbidden_input_fields=("reward_surrogate",),
        )


def test_stage23_active_registry_contains_only_promoted_sampler() -> None:
    active = build_active_policy_gradient_sampler_registry()
    trial = build_stage23_trial_sampler_registry()

    assert set(trial) == {
        PHYSICAL_BERNOULLI_PROPOSAL_SAMPLER_ID,
        PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID,
        ENDPOINT_BUDGETED_PHYSICAL_PROPOSAL_SAMPLER_ID,
    }
    assert set(active) == {ACTIVE_POLICY_GRADIENT_SAMPLER_ID}
    assert ACTIVE_POLICY_GRADIENT_SAMPLER_ID == PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID
