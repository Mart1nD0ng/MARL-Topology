from marl_topology.training.policy_gradient.samplers import (
    ACTIVE_POLICY_GRADIENT_SAMPLER_ID,
    ARCHIVED_TRIAL_SAMPLER_IDS,
    build_active_policy_gradient_sampler_registry,
    sampler_cleanup_report,
)


def test_stage23_active_sampler_registry_is_low_entropy_after_promotion() -> None:
    registry = build_active_policy_gradient_sampler_registry()
    cleanup = sampler_cleanup_report()

    assert list(registry) == [ACTIVE_POLICY_GRADIENT_SAMPLER_ID]
    assert cleanup["active_sampler_count"] == 1
    assert cleanup["active_sampler_ids"] == [ACTIVE_POLICY_GRADIENT_SAMPLER_ID]
    assert cleanup["losing_samplers_removed_from_active_registry"] is True
    for archived_id in ARCHIVED_TRIAL_SAMPLER_IDS:
        assert archived_id not in registry
