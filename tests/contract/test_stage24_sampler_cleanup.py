from marl_topology.training.mappo.trainer import run_stage24_critic_integrated_micro_loop
from marl_topology.training.policy_gradient.samplers import (
    ACTIVE_POLICY_GRADIENT_SAMPLER_ID,
    ENDPOINT_BUDGETED_PHYSICAL_PROPOSAL_SAMPLER_ID,
    PHYSICAL_BERNOULLI_PROPOSAL_SAMPLER_ID,
    PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID,
    build_active_policy_gradient_sampler_registry,
    build_stage24_candidate_sampler_registry,
    sampler_cleanup_report,
)


def test_stage24_candidate_registry_compares_plackett_luce_and_endpoint_only() -> None:
    candidates = build_stage24_candidate_sampler_registry()
    active = build_active_policy_gradient_sampler_registry()

    assert set(candidates) == {
        PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID,
        ENDPOINT_BUDGETED_PHYSICAL_PROPOSAL_SAMPLER_ID,
    }
    assert PHYSICAL_BERNOULLI_PROPOSAL_SAMPLER_ID not in candidates
    assert list(active) == [ACTIVE_POLICY_GRADIENT_SAMPLER_ID]
    assert ACTIVE_POLICY_GRADIENT_SAMPLER_ID == PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID


def test_stage24_micro_closeout_leaves_exactly_one_active_sampler() -> None:
    report = run_stage24_critic_integrated_micro_loop(mode="smoke", project_root=".")
    cleanup = sampler_cleanup_report()

    # Budget fix (rsu 4->8) consequence: the budget-aware endpoint sampler now wins
    # the stage24 A/B comparison (tau 0.5625 vs PL 0.125 under rsu=8), so the live
    # winner != the registered active sampler and the loop flags owner_decision.
    # Owner kept Plackett-Luce active (the endpoint sampler over-selects under
    # route B; variable proposal size is the real fix, deferred), so the active
    # registry/cleanup below is unchanged.
    assert report["sampler_selection"]["selected_sampler_id"] == (
        ENDPOINT_BUDGETED_PHYSICAL_PROPOSAL_SAMPLER_ID
    )
    assert report["owner_decision_required"] is True
    assert cleanup["active_sampler_count"] == 1
    assert cleanup["active_sampler_ids"] == [PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID]
    assert ENDPOINT_BUDGETED_PHYSICAL_PROPOSAL_SAMPLER_ID not in (
        cleanup["active_sampler_ids"]
    )
    assert cleanup["losing_samplers_removed_from_active_registry"] is True
