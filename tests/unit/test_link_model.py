from marl_topology.link import (
    STAGE2_DETERMINISTIC_DISTANCE_REGIME,
    STAGE2_DETERMINISTIC_DISTANCE_REGIME_ID,
    SimpleLinkModel,
    validate_link_record_against_regime,
)
from marl_topology.topology import CandidateEdge
from marl_topology.scenario import make_demo_scene
from marl_topology.topology import CandidateGraph


def test_link_reliability_monotonic_with_distance() -> None:
    model = SimpleLinkModel(reference_distance_m=100.0)

    near = model.link_success_probability(10.0)
    far = model.link_success_probability(80.0)

    assert 0.0 <= far < near <= 1.0


def test_link_latency_and_energy_are_nonnegative() -> None:
    graph = CandidateGraph.from_scene(make_demo_scene(), max_distance_m=80.0)
    records = SimpleLinkModel().evaluate_graph(graph)

    assert records
    assert all(record.latency_s >= 0.0 for record in records.values())
    assert all(record.energy_j >= 0.0 for record in records.values())
    assert {record.physics_regime for record in records.values()} == {
        "stage2_deterministic_distance"
    }


def test_stage2_link_regime_metadata_declares_boundary() -> None:
    regime = STAGE2_DETERMINISTIC_DISTANCE_REGIME

    assert regime.regime_id == STAGE2_DETERMINISTIC_DISTANCE_REGIME_ID
    assert regime.units["distance_3d_m"] == "m"
    assert regime.units["link_success_probability"] == "[0, 1]"
    assert "SINR" in regime.omitted_components
    assert "interference" in regime.omitted_components
    assert "not a reward definition" in regime.nonclaims


def test_link_record_validates_against_stage2_regime() -> None:
    edge = CandidateEdge("a--b", "a", "b", 25.0)
    record = SimpleLinkModel().evaluate_edge(edge, STAGE2_DETERMINISTIC_DISTANCE_REGIME_ID)

    validate_link_record_against_regime(record)

    bad_record = type(
        "BadRecord",
        (),
        {
            "edge_id": "a--b",
            "distance_3d_m": 25.0,
            "link_success_probability": 0.5,
            "latency_s": 0.001,
            "energy_j": 0.1,
            "physics_regime": "unreviewed_regime",
        },
    )()
    try:
        validate_link_record_against_regime(bad_record)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong physics regime accepted")


def test_latency_and_energy_are_monotonic_with_distance() -> None:
    model = SimpleLinkModel(reference_distance_m=100.0)
    near = model.evaluate_edge(
        CandidateEdge("a--b", "a", "b", 10.0),
        STAGE2_DETERMINISTIC_DISTANCE_REGIME_ID,
    )
    far = model.evaluate_edge(
        CandidateEdge("a--c", "a", "c", 80.0),
        STAGE2_DETERMINISTIC_DISTANCE_REGIME_ID,
    )

    assert near.link_success_probability > far.link_success_probability
    assert near.latency_s < far.latency_s
    assert near.energy_j < far.energy_j
