"""Stage 31 Phase E: production dataset, leakage-checked split, scalable teacher."""

from __future__ import annotations

import pytest

from marl_topology.data.stage31_production_dataset import (
    build_dataset_manifest,
    build_production_context,
    build_production_dataset,
    split_scenarios,
)
from marl_topology.data.stage31_scenario_generator import (
    ProductionScenarioConfig,
    generate_production_scenarios,
)

TAU = 0.9


@pytest.fixture(scope="module")
def dataset():
    return build_production_dataset(
        ProductionScenarioConfig(seed=31, scenario_count=100)
    )


def test_split_has_no_cross_split_leakage(dataset) -> None:
    train = set(dataset.split.train_scenario_ids)
    eval_ids = set(dataset.split.eval_scenario_ids)
    test = set(dataset.split.test_scenario_ids)
    assert train and eval_ids and test
    assert not (train & eval_ids)
    assert not (train & test)
    assert not (eval_ids & test)
    assert dataset.quality_report["split_leakage_overlap_count"] == 0


def test_every_split_contains_feasible_and_infeasible(dataset) -> None:
    assert dataset.quality_report["each_split_has_feasible_and_infeasible"] is True


def test_dataset_is_production_scale_ready(dataset) -> None:
    qr = dataset.quality_report
    assert qr["scenario_count"] >= 100
    assert qr["duplicate_context_rate"] == 0.0
    assert qr["production_scale_ready"] is True


def test_teacher_labels_are_feasible_and_not_actor_inputs(dataset) -> None:
    for spec in dataset.specs:
        label = dataset.teacher_labels[spec.scenario_id]
        assert label["is_actor_input"] is False
        if spec.feasible_exists:
            assert label["feasible_exists"] is True
            assert label["consensus_success_probability"] >= TAU
            # The teacher topology is sparse (never the full graph under interference).
            assert label["selected_edge_count"] <= spec.candidate_edge_count


def test_split_is_deterministic() -> None:
    specs = generate_production_scenarios(
        ProductionScenarioConfig(seed=9, scenario_count=40)
    )
    a = split_scenarios(specs, seed=9)
    b = split_scenarios(specs, seed=9)
    assert a.train_scenario_ids == b.train_scenario_ids
    assert a.eval_scenario_ids == b.eval_scenario_ids
    assert a.test_scenario_ids == b.test_scenario_ids


def test_production_context_round_trips(dataset) -> None:
    spec = dataset.specs[0]
    ctx = build_production_context(spec)
    assert ctx.fixture.fixture_id == spec.scenario_id
    assert set(ctx.graph.edge_ids)  # has candidate edges
    # The evaluator reproduces the measured feasibility for the full graph.
    from marl_topology.policies import PolicyBaselines

    full = PolicyBaselines.full(ctx.graph)
    ev = ctx.evaluator.evaluate(set(full.edge_ids))
    assert float(ev.metrics["consensus_success_probability"]) == pytest.approx(
        spec.full_graph_psucc, abs=1e-9
    )


def test_manifest_has_required_fields(dataset) -> None:
    manifest = build_dataset_manifest(dataset)
    for key in (
        "dataset_id",
        "physics_regime_id",
        "protocol_model_id",
        "tau_requirement_min",
        "teacher_label_is_actor_input",
        "owner_approval_id",
    ):
        assert key in manifest
    assert manifest["tau_requirement_min"] == TAU
    assert manifest["teacher_label_is_actor_input"] is False
