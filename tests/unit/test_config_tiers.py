"""R0 (v2 Engineering-Plan §R0): config tiers smoke / pilot / research.

A research config must NOT inherit smoke params, and smoke/pilot results must be
structurally barred from becoming a headline (v2 plan exit condition + D4). These
tests pin the tier schema, the on-disk tiers, and the headline-eligibility guard.
"""

import pytest

from marl_topology.training.config_tiers import (
    HeadlineEligibilityError,
    TIER_NAMES,
    assert_headline_eligible,
    load_tier_config,
    validate_tier_config,
)


def test_three_tiers_exist_on_disk_and_validate():
    assert TIER_NAMES == ("smoke", "pilot", "research")
    for tier in TIER_NAMES:
        config = load_tier_config(tier)  # loads + validates configs/<tier>/default.json
        assert config["tier"] == tier


def test_smoke_and_pilot_are_headline_ineligible_research_is_eligible():
    assert load_tier_config("smoke")["headline_eligible"] is False
    assert load_tier_config("pilot")["headline_eligible"] is False
    assert load_tier_config("research")["headline_eligible"] is True


def test_research_does_not_inherit_smoke_params():
    smoke = load_tier_config("smoke")
    research = load_tier_config("research")
    # research is a genuinely different, heavier regime -- not a relabeled smoke
    assert len(research["seeds"]) >= 5            # D4: >=5 seeds for a headline
    assert len(smoke["seeds"]) < len(research["seeds"])
    assert research["updates"] > smoke["updates"]
    assert research["scene_count"] > smoke["scene_count"]


def test_headline_guard_refuses_smoke_and_pilot():
    for tier in ("smoke", "pilot"):
        with pytest.raises(HeadlineEligibilityError):
            assert_headline_eligible(load_tier_config(tier))
    # research passes the guard
    assert_headline_eligible(load_tier_config("research"))


def test_schema_rejects_unknown_tier():
    with pytest.raises(ValueError):
        validate_tier_config({"tier": "production", "headline_eligible": True, "purpose": "x",
                              "updates": 1, "scene_count": 1, "val_scene_count": 1,
                              "seeds": [0], "samples_per_scene": 1})


def test_schema_rejects_missing_required_field():
    with pytest.raises(ValueError):
        validate_tier_config({"tier": "smoke"})


def test_schema_rejects_research_with_too_few_seeds():
    bad = {"tier": "research", "headline_eligible": True, "purpose": "x", "updates": 100,
           "scene_count": 100, "val_scene_count": 10, "seeds": [0, 1], "samples_per_scene": 1}
    with pytest.raises(ValueError):
        validate_tier_config(bad)


def test_schema_rejects_smoke_marked_headline_eligible():
    bad = {"tier": "smoke", "headline_eligible": True, "purpose": "x", "updates": 1,
           "scene_count": 1, "val_scene_count": 1, "seeds": [0], "samples_per_scene": 1}
    with pytest.raises(ValueError):
        validate_tier_config(bad)
