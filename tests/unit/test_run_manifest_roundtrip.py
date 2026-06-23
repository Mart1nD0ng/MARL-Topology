"""R0 (v2 Engineering-Plan §R0): the active run manifest + arms registry.

A fresh run must be reproducible from its manifest (git revision, env-math version,
action-distribution version, dataset manifest, seed/split, mechanism activation,
evaluator-call budget). This is a NEW *active* manifest, distinct from the frozen
Stage-5.9/5.10 dry-run design contract. The arms registry pins the four comparison
arms (EMA / RLOO / Graph-MAPPO / production) with their evaluator-call budgets.
"""

import pytest

from marl_topology.training.run_manifest import (
    ARMS,
    REQUIRED_MANIFEST_FIELDS,
    RunManifest,
    build_run_manifest,
)


def _manifest(**overrides):
    base = dict(
        git_revision="abc1234",
        seed=0,
        split="train",
        dataset_manifest={"shards": [9101, 9102], "scene_count": 72, "feasible_count": 50},
        mechanism_activation={"baseline": "ema", "sampler": "bcsp", "fault_model": "fixed_set"},
        evaluator_call_budget=1,
        tier="research",
        arm="ema",
        created_at_utc="2026-06-23T00:00:00Z",
        headline_eligible=True,
    )
    base.update(overrides)
    return build_run_manifest(**base)


def test_required_v2_fields_present():
    m = _manifest().to_dict()
    for field in ("git_revision", "environment_math_version", "action_distribution_version",
                  "dataset_manifest", "seed", "split", "mechanism_activation",
                  "evaluator_call_budget"):
        assert field in m and m[field] not in (None, "", {})
    assert set(REQUIRED_MANIFEST_FIELDS) <= set(m)


def test_to_dict_from_dict_roundtrip():
    m = _manifest()
    assert RunManifest.from_dict(m.to_dict()) == m


def test_write_read_roundtrip(tmp_path):
    m = _manifest()
    path = tmp_path / "run_manifest.json"
    m.write(path)
    assert RunManifest.read(path) == m


def test_mechanism_activation_is_present_and_nonempty():
    # D6: every run records which mechanisms it activated
    assert isinstance(_manifest().mechanism_activation, dict)
    assert _manifest().mechanism_activation


def test_build_stamps_env_and_action_versions():
    m = _manifest()
    assert m.environment_math_version  # stamped by build_run_manifest, not caller-supplied
    assert m.action_distribution_version


def test_arms_registry_has_four_arms_with_budgets():
    assert set(ARMS) == {"ema", "rloo", "graph-mappo", "production"}
    for arm in ARMS.values():
        assert arm["evaluator_calls_per_scene"] >= 1
    assert ARMS["ema"]["evaluator_calls_per_scene"] == 1
    assert ARMS["graph-mappo"]["evaluator_calls_per_scene"] == 1
    assert ARMS["rloo"]["evaluator_calls_per_scene"] >= 2


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError):
        _manifest(arm="not-an-arm")


def test_smoke_tier_cannot_be_headline_eligible():
    with pytest.raises(ValueError):
        _manifest(tier="smoke", headline_eligible=True)
