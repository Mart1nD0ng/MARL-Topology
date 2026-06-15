"""Deterministic scenario fixtures for Stage 2 contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from marl_topology.geometry3d import Point3D
from marl_topology.scenario.scene import Node3D, NodeKind, Scene3D, make_demo_scene


@dataclass(frozen=True, slots=True)
class ScenarioFixture:
    fixture_id: str
    description: str
    scene: Scene3D
    max_candidate_distance_m: float
    quorum_size: int
    success_probability_threshold: float
    deadline_s: float | None
    link_reference_distance_m: float
    expected_oracle_status: str
    expected_full_graph_success: bool

    def __post_init__(self) -> None:
        if self.fixture_id != self.scene.scenario_id:
            raise ValueError("fixture_id must match scene.scenario_id")
        if self.max_candidate_distance_m < 0:
            raise ValueError("max_candidate_distance_m must be nonnegative")
        if self.quorum_size <= 0:
            raise ValueError("quorum_size must be positive")
        if self.quorum_size > len(self.scene.nodes):
            raise ValueError("quorum_size cannot exceed fixture node count")
        if not 0.0 <= self.success_probability_threshold <= 1.0:
            raise ValueError("success_probability_threshold must be in [0, 1]")
        if self.deadline_s is not None and self.deadline_s < 0:
            raise ValueError("deadline_s must be nonnegative")
        if self.link_reference_distance_m <= 0:
            raise ValueError("link_reference_distance_m must be positive")
        if self.expected_oracle_status not in {"feasible", "infeasible", "unresolved"}:
            raise ValueError("expected_oracle_status must be feasible, infeasible, or unresolved")
        if self.scene.physics_regime != "stage2_deterministic_distance":
            raise ValueError("Stage 2.5 fixtures must use stage2_deterministic_distance")


def iter_scenario_fixtures() -> tuple[ScenarioFixture, ...]:
    return (
        demo_stage2_fixture(),
        sparse_chain_stage2_fixture(),
        quorum_blocked_stage2_fixture(),
    )


def get_scenario_fixture(fixture_id: str) -> ScenarioFixture:
    for fixture in iter_scenario_fixtures():
        if fixture.fixture_id == fixture_id:
            return fixture
    raise KeyError(f"unknown scenario fixture: {fixture_id}")


def demo_stage2_fixture() -> ScenarioFixture:
    return ScenarioFixture(
        fixture_id="demo_stage2",
        description="Dense four-node smoke scene used by the Stage 2 demo.",
        scene=make_demo_scene(),
        max_candidate_distance_m=80.0,
        quorum_size=3,
        success_probability_threshold=0.45,
        deadline_s=0.01,
        link_reference_distance_m=120.0,
        expected_oracle_status="feasible",
        expected_full_graph_success=True,
    )


def sparse_chain_stage2_fixture() -> ScenarioFixture:
    return ScenarioFixture(
        fixture_id="sparse_chain_stage2",
        description="Four-node chain where a sparse quorum path is feasible.",
        scene=_scene(
            "sparse_chain_stage2",
            (
                Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 5.0)),
                Node3D("veh_0", NodeKind.VEHICLE, Point3D(35.0, 0.0, 1.5)),
                Node3D("veh_1", NodeKind.VEHICLE, Point3D(70.0, 0.0, 1.5)),
                Node3D("veh_2", NodeKind.VEHICLE, Point3D(105.0, 0.0, 1.5)),
            ),
        ),
        max_candidate_distance_m=45.0,
        quorum_size=3,
        success_probability_threshold=0.45,
        deadline_s=0.01,
        link_reference_distance_m=120.0,
        expected_oracle_status="feasible",
        expected_full_graph_success=True,
    )


def quorum_blocked_stage2_fixture() -> ScenarioFixture:
    return ScenarioFixture(
        fixture_id="quorum_blocked_stage2",
        description="Quorum-unreachable scene where policy failure is not confused with oracle evidence.",
        scene=_scene(
            "quorum_blocked_stage2",
            (
                Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 5.0)),
                Node3D("veh_0", NodeKind.VEHICLE, Point3D(25.0, 0.0, 1.5)),
                Node3D("veh_1", NodeKind.VEHICLE, Point3D(200.0, 0.0, 1.5)),
                Node3D("veh_2", NodeKind.VEHICLE, Point3D(260.0, 0.0, 1.5)),
            ),
        ),
        max_candidate_distance_m=40.0,
        quorum_size=3,
        success_probability_threshold=0.45,
        deadline_s=0.01,
        link_reference_distance_m=120.0,
        expected_oracle_status="infeasible",
        expected_full_graph_success=False,
    )


def _scene(scenario_id: str, nodes: Iterable[Node3D]) -> Scene3D:
    return Scene3D(scenario_id=scenario_id, nodes=tuple(nodes))
