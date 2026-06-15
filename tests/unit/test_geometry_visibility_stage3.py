from marl_topology.geometry3d import (
    BuildingBox,
    LoSState,
    Point3D,
    distance_3d,
    evaluate_visibility,
    ray_box_intersection,
)
from marl_topology.scenario import (
    get_geometry_visibility_fixture,
    iter_geometry_visibility_fixtures,
)


def test_3d_distance_uses_z_axis() -> None:
    assert distance_3d(Point3D(0.0, 0.0, 0.0), Point3D(3.0, 4.0, 12.0)) == 13.0


def test_ray_box_intersection_records_entry_and_exit_distances() -> None:
    box = BuildingBox.from_footprint_height(
        "blocker",
        min_x_m=4.0,
        max_x_m=6.0,
        min_y_m=-1.0,
        max_y_m=1.0,
        height_m=5.0,
    )
    hit = ray_box_intersection(Point3D(0.0, 0.0, 2.0), Point3D(10.0, 0.0, 2.0), box)

    assert hit is not None
    assert hit.building_id == "blocker"
    assert hit.entry_distance_m == 4.0
    assert hit.exit_distance_m == 6.0


def test_stage3_geometry_fixtures_match_expected_visibility() -> None:
    for fixture in iter_geometry_visibility_fixtures():
        record = evaluate_visibility(fixture.scene, fixture.tx_id, fixture.rx_id)

        assert record.scenario_id == fixture.fixture_id
        assert record.distance_3d_m >= 0.0
        assert record.los_state.value == fixture.expected_los_state
        assert tuple(blocker.building_id for blocker in record.blocker_records) == (
            fixture.expected_blocker_ids
        )


def test_building_height_can_change_los_state() -> None:
    high_fixture = get_geometry_visibility_fixture("blocked_by_building")
    high_record = evaluate_visibility(
        high_fixture.scene,
        high_fixture.tx_id,
        high_fixture.rx_id,
    )
    assert high_record.los_state == LoSState.NLOS

    low_building = BuildingBox.from_footprint_height(
        "building_blocker",
        min_x_m=40.0,
        max_x_m=60.0,
        min_y_m=-10.0,
        max_y_m=10.0,
        height_m=1.0,
    )
    low_scene = type(high_fixture.scene)(
        scenario_id="height_change_los",
        nodes=high_fixture.scene.nodes,
        buildings=(low_building,),
        roads=high_fixture.scene.roads,
        lanes=high_fixture.scene.lanes,
        physics_regime=high_fixture.scene.physics_regime,
    )
    low_record = evaluate_visibility(low_scene, high_fixture.tx_id, high_fixture.rx_id)

    assert low_record.los_state == LoSState.LOS
    assert low_record.blocker_records == ()


def test_urban_gap_preserves_los_corridor() -> None:
    fixture = get_geometry_visibility_fixture("urban_gap_los")
    record = evaluate_visibility(fixture.scene, fixture.tx_id, fixture.rx_id)

    assert record.los_state == LoSState.LOS
    assert record.blocker_records == ()
    assert {building.building_id for building in fixture.scene.buildings} == {
        "north_wall",
        "south_wall",
    }


def test_rsu_height_can_restore_los() -> None:
    low_fixture = get_geometry_visibility_fixture("rsu_low_nlos")
    high_fixture = get_geometry_visibility_fixture("rsu_high_los")

    low_record = evaluate_visibility(low_fixture.scene, low_fixture.tx_id, low_fixture.rx_id)
    high_record = evaluate_visibility(high_fixture.scene, high_fixture.tx_id, high_fixture.rx_id)

    assert low_record.los_state == LoSState.NLOS
    assert tuple(blocker.building_id for blocker in low_record.blocker_records) == ("low_blocker",)
    assert high_record.los_state == LoSState.LOS
    assert high_record.blocker_records == ()


def test_blocker_record_order_is_stable_by_entry_distance_then_id() -> None:
    fixture = get_geometry_visibility_fixture("urban_canyon_nlos")
    record = evaluate_visibility(fixture.scene, fixture.tx_id, fixture.rx_id)
    entries = [
        (blocker.entry_distance_m, blocker.building_id)
        for blocker in record.blocker_records
    ]

    assert entries == sorted(entries)
