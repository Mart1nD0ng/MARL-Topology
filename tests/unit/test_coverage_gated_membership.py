"""Contract tests for coverage-gated PBFT validator membership (Stage 21 evaluator).

Membership is a SCENE property: a node whose best incident candidate link delivery is
below the floor cannot complete a primary round under any topology, so it is demoted to
a client and consensus is evaluated over the covered validator set. The policy can never
game membership because it is fixed before any topology is selected.
"""

from marl_topology.data.stage21_objective_stack_evidence import (
    Stage21ObjectiveStackConfig,
    Stage21ObjectiveStackEvaluator,
)
from marl_topology.data.stage31_scenario_generator import PhysicsRegime, build_stack_config
from marl_topology.geometry3d import Point3D
from marl_topology.scenario.scene import Node3D, NodeKind, Scene3D
from marl_topology.topology import CandidateGraph


FAR_AWAY_M = 100_000.0


def _clean_regime(**kwargs) -> PhysicsRegime:
    return PhysicsRegime(
        tx_power_dbm=20.0,
        use_background_interference=False,
        orthogonal_resources=True,
        **kwargs,
    )


def _cluster_plus_far_scene(cluster_count: int, far_count: int) -> Scene3D:
    nodes = [
        Node3D(f"veh_{i}", NodeKind.VEHICLE, Point3D(20.0 * (i % 2), 20.0 * (i // 2), 1.5))
        for i in range(cluster_count)
    ]
    nodes.extend(
        Node3D(
            f"veh_far_{j}",
            NodeKind.VEHICLE,
            Point3D(FAR_AWAY_M + 50_000.0 * j, FAR_AWAY_M, 1.5),
        )
        for j in range(far_count)
    )
    return Scene3D(scenario_id="membership_test", nodes=tuple(nodes))


def _evaluator(scene: Scene3D, config: Stage21ObjectiveStackConfig) -> Stage21ObjectiveStackEvaluator:
    graph = CandidateGraph.from_scene(scene, max_distance_m=None)
    return Stage21ObjectiveStackEvaluator(scene=scene, graph=graph, config=config)


def test_default_off_keeps_all_nodes_and_metrics_unchanged() -> None:
    scene = _cluster_plus_far_scene(4, 1)
    config = build_stack_config(_clean_regime())
    assert config.coverage_gated_membership is False
    evaluator = _evaluator(scene, config)
    assert evaluator.validator_ids == evaluator.graph.node_ids
    metrics = evaluator.evaluate(set(evaluator.graph.edge_ids)).metrics
    assert "coverage_rate" not in metrics
    assert "validator_count" not in metrics


def test_gating_lifts_certified_infeasible_scene() -> None:
    # 4 strong cluster nodes + 1 unreachable node: ungated expected-initiator consensus is
    # capped at (N - 1)/N = 0.8 < tau; gating demotes the dead node and the cluster passes.
    scene = _cluster_plus_far_scene(4, 1)
    ungated = _evaluator(scene, build_stack_config(_clean_regime()))
    gated = _evaluator(
        scene, build_stack_config(_clean_regime(coverage_gated_membership=True))
    )
    selected = set(gated.graph.edge_ids)
    p_ungated = float(ungated.evaluate(selected).metrics["consensus_success_probability"])
    gated_metrics = gated.evaluate(selected).metrics
    p_gated = float(gated_metrics["consensus_success_probability"])
    assert p_ungated <= 0.8 + 1e-9
    assert p_gated >= 0.9
    assert gated.validator_ids == ("veh_0", "veh_1", "veh_2", "veh_3")
    assert gated_metrics["validator_count"] == 4
    assert abs(float(gated_metrics["coverage_rate"]) - 0.8) < 1e-12


def test_membership_is_topology_independent() -> None:
    scene = _cluster_plus_far_scene(4, 1)
    gated = _evaluator(
        scene, build_stack_config(_clean_regime(coverage_gated_membership=True))
    )
    empty = gated.evaluate(set())
    full = gated.evaluate(set(gated.graph.edge_ids))
    assert empty.metrics["validator_count"] == full.metrics["validator_count"]
    assert empty.metrics["coverage_rate"] == full.metrics["coverage_rate"]


def test_fewer_than_four_validators_is_infeasible() -> None:
    scene = _cluster_plus_far_scene(3, 2)
    gated = _evaluator(
        scene, build_stack_config(_clean_regime(coverage_gated_membership=True))
    )
    assert len(gated.validator_ids) == 3
    evaluation = gated.evaluate(set(gated.graph.edge_ids))
    assert float(evaluation.metrics["consensus_success_probability"]) == 0.0
    assert evaluation.per_primary_reliability == {"veh_0": 0.0, "veh_1": 0.0, "veh_2": 0.0}


def test_wired_backhaul_keeps_remote_rsus_as_validators() -> None:
    nodes = tuple(
        [
            Node3D(f"veh_{i}", NodeKind.VEHICLE, Point3D(20.0 * (i % 2), 20.0 * (i // 2), 1.5))
            for i in range(4)
        ]
        + [
            Node3D("rsu_0", NodeKind.RSU, Point3D(FAR_AWAY_M, 0.0, 5.0)),
            Node3D("rsu_1", NodeKind.RSU, Point3D(0.0, FAR_AWAY_M, 5.0)),
        ]
    )
    scene = Scene3D(scenario_id="membership_rsu_test", nodes=nodes)
    regime = _clean_regime(coverage_gated_membership=True, wired_rsu_backhaul=True)
    gated = _evaluator(scene, build_stack_config(regime))
    assert "rsu_0" in gated.validator_ids
    assert "rsu_1" in gated.validator_ids
    no_backhaul = _evaluator(
        scene, build_stack_config(_clean_regime(coverage_gated_membership=True))
    )
    assert "rsu_0" not in no_backhaul.validator_ids
