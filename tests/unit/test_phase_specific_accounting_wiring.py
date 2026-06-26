"""D4: phase-specific PBFT message-plan accounting wired into the Stage-21 evaluator (Spec S4.8-4.10).

The legacy evaluator reused ONE all-pairs record set for all three PBFT phases
(``phase_records = {pre: records, prepare: records, commit: records}``), so the energy accounting
counted the pre-prepare round as full all-pairs x3 (it should be a primary->backups STAR) and could
include client links. With ``phase_specific_accounting=True`` the ENERGY/LATENCY accounting uses the
phase-specific plan (pre_prepare = primary star, prepare/commit = validator<->validator, clients never
vote). The RELIABILITY matrices are untouched (the expected-initiator model averages over all primaries
internally, so its matrices must stay the full validator set). Default off => byte-identical.
"""

from __future__ import annotations

import dataclasses

import pytest

from marl_topology.data.stage21_objective_stack_evidence import (
    Stage21ObjectiveStackConfig,
    Stage21ObjectiveStackEvaluator,
)
from marl_topology.data.stage31_production_dataset import build_scenario_evaluator
from marl_topology.data.stage31_scenario_generator import (
    PhysicsRegime,
    ProductionScenarioConfig,
    build_stack_config,
    generate_production_scenarios,
)
from marl_topology.geometry3d import Point3D
from marl_topology.scenario.scene import Node3D, NodeKind, Scene3D
from marl_topology.topology import CandidateGraph


def _all_validator_scene():
    spec = generate_production_scenarios(
        ProductionScenarioConfig(seed=5, scenario_count=1, node_count_choices=(8,))
    )[0]
    graph, canonical = build_scenario_evaluator(spec)
    return spec, graph, canonical


def _corrected(canonical, scene, graph):
    return Stage21ObjectiveStackEvaluator(
        scene=scene, graph=graph,
        config=dataclasses.replace(canonical.config, phase_specific_accounting=True))


def test_phase_accounting_off_byte_identical() -> None:
    spec, graph, canonical = _all_validator_scene()
    again = Stage21ObjectiveStackEvaluator(scene=spec.scene, graph=graph, config=canonical.config)
    full = set(graph.edge_ids)
    a, b = again.evaluate(full).metrics, canonical.evaluate(full).metrics
    assert a["energy"] == pytest.approx(b["energy"])
    assert a["latency"] == pytest.approx(b["latency"])
    assert "energy_breakdown" not in a            # the new key is absent when the flag is off


def test_pre_prepare_is_star_not_all_pairs() -> None:
    spec, graph, canonical = _all_validator_scene()
    corrected = _corrected(canonical, spec.scene, graph)
    n_val = len(corrected.validator_ids)
    bd = corrected.evaluate(set(graph.edge_ids)).metrics["energy_breakdown"]
    # pre-prepare is the primary -> backups STAR: exactly (n_val - 1) messages, strictly fewer than the
    # validator<->validator all-pairs count used by prepare/commit.
    assert bd["pre_prepare_messages"] == n_val - 1
    assert bd["prepare_messages"] == n_val * (n_val - 1)
    assert bd["pre_prepare_messages"] < bd["prepare_messages"]
    assert bd["pre_prepare_energy"] <= bd["prepare_energy"] + 1e-12


def test_phase_specific_energy_below_all_pairs_x3() -> None:
    spec, graph, canonical = _all_validator_scene()
    corrected = _corrected(canonical, spec.scene, graph)
    full = set(graph.edge_ids)
    legacy_e = canonical.evaluate(full).metrics["energy"]            # all-pairs x3 (over-counts pre-prepare)
    corrected_e = corrected.evaluate(full).metrics["energy"]
    assert corrected_e < legacy_e                                   # the over-count is removed
    assert corrected_e > 0.0


def test_reliability_unchanged_by_flag() -> None:
    spec, graph, canonical = _all_validator_scene()
    corrected = _corrected(canonical, spec.scene, graph)
    full = set(graph.edge_ids)
    # the phase-specific change touches ONLY energy/latency accounting; reliability matrices are full.
    assert corrected.evaluate(full).metrics["consensus_success_probability"] == pytest.approx(
        canonical.evaluate(full).metrics["consensus_success_probability"])


def _cluster_plus_far_scene(cluster_count: int, far_count: int) -> Scene3D:
    nodes = [Node3D(f"veh_{i}", NodeKind.VEHICLE, Point3D(20.0 * (i % 2), 20.0 * (i // 2), 1.5))
             for i in range(cluster_count)]
    nodes.extend(Node3D(f"veh_far_{j}", NodeKind.VEHICLE, Point3D(100_000.0 + 50_000.0 * j, 100_000.0, 1.5))
                 for j in range(far_count))
    return Scene3D(scenario_id="d4_clients", nodes=tuple(nodes))


def test_clients_do_not_vote() -> None:
    scene = _cluster_plus_far_scene(5, 2)
    graph = CandidateGraph.from_scene(scene, max_distance_m=None)
    config = build_stack_config(PhysicsRegime(
        tx_power_dbm=20.0, use_background_interference=False, orthogonal_resources=True,
        coverage_gated_membership=True))
    config = dataclasses.replace(config, phase_specific_accounting=True)
    ev = Stage21ObjectiveStackEvaluator(scene=scene, graph=graph, config=config)
    n_val = len(ev.validator_ids)
    bd = ev.evaluate(set(graph.edge_ids)).metrics["energy_breakdown"]
    assert bd["client_count"] >= 1                                  # the far nodes are demoted to clients
    assert n_val < len(graph.node_ids)                             # so validators are a strict subset
    # prepare/commit are validator<->validator ONLY: exactly n_val*(n_val-1) messages, clients excluded.
    assert bd["prepare_messages"] == n_val * (n_val - 1)
    assert bd["commit_messages"] == n_val * (n_val - 1)
    assert bd["pre_prepare_messages"] == n_val - 1


def test_vectorized_matches_canonical_with_phase_specific() -> None:
    # BLOCKER guard: the vectorized fast-path evaluator (the dynamic arm's default, vectorized=True)
    # MUST also apply phase-specific accounting, or it silently keeps the all-pairs-x3 energy and
    # diverges from the canonical evaluator. Fails on the canonical-only D4 (vectorized ignored the flag).
    from marl_topology.data.vectorized_objective_stack_evaluator import VectorizedStage21Evaluator
    spec, graph, canonical = _all_validator_scene()
    cfg = dataclasses.replace(canonical.config, phase_specific_accounting=True)
    can = Stage21ObjectiveStackEvaluator(scene=spec.scene, graph=graph, config=cfg)
    fast = VectorizedStage21Evaluator(scene=spec.scene, graph=graph, config=cfg, ref=can)
    full = set(graph.edge_ids)
    cm, fm = can.evaluate(full).metrics, fast.evaluate(full).metrics
    assert abs(cm["energy"] - fm["energy"]) < 1e-9          # phase-specific energy float-identical
    assert abs(cm["latency"] - fm["latency"]) < 1e-9
    assert abs(cm["consensus_success_probability"] - fm["consensus_success_probability"]) < 1e-9
    assert "energy_breakdown" in fm                          # the fast path exposes the breakdown too
    assert fm["energy_breakdown"]["pre_prepare_messages"] == cm["energy_breakdown"]["pre_prepare_messages"]


def test_reliability_unchanged_by_flag_fixed_set() -> None:
    # The production operating point uses the fixed_set fault model -> verify reliability is identical
    # on<->off there too (not just remove_largest). Reliability must never see the phase restriction.
    spec, graph, canonical = _all_validator_scene()
    base = dataclasses.replace(canonical.config, fault_model="fixed_set")
    off = Stage21ObjectiveStackEvaluator(scene=spec.scene, graph=graph, config=base)
    on = Stage21ObjectiveStackEvaluator(
        scene=spec.scene, graph=graph,
        config=dataclasses.replace(base, phase_specific_accounting=True))
    full = set(graph.edge_ids)
    assert on.evaluate(full).metrics["consensus_success_probability"] == pytest.approx(
        off.evaluate(full).metrics["consensus_success_probability"])
    # the per-primary reliability vector must be identical too (reliability matrices are full)
    assert dict(on.evaluate(full).per_primary_reliability) == pytest.approx(
        dict(off.evaluate(full).per_primary_reliability))


def test_operating_point_regime_activates_phase_specific_accounting() -> None:
    # The production/dynamic regime must ACTIVATE the corrected accounting (not leave it inert).
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "bop_d4", root / "scripts" / "train" / "build_operating_point_dataset.py")
    bop = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0, str(root / "scripts" / "train"))
    spec.loader.exec_module(bop)
    regime = bop.operating_point_regime(20.0)
    assert getattr(regime, "phase_specific_accounting", False) is True
    cfg = build_stack_config(regime)
    assert cfg.phase_specific_accounting is True
