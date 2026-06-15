from pathlib import Path

from marl_topology.geometry3d import BlockerRecord, LoSState, VisibilityRecord
from marl_topology.scenario import Lane, RoadSegment, get_geometry_visibility_fixture


ROOT = Path(__file__).resolve().parents[2]


def test_geometry_visibility_public_interfaces_exist() -> None:
    fixture = get_geometry_visibility_fixture("free_space_close")

    assert fixture.scene.roads
    assert fixture.scene.lanes
    assert isinstance(fixture.scene.roads[0], RoadSegment)
    assert isinstance(fixture.scene.lanes[0], Lane)
    assert fixture.expected_los_state == "los"
    assert LoSState.LOS.value == "los"
    assert VisibilityRecord
    assert BlockerRecord


def test_stage3_geometry_contract_records_implementation_status() -> None:
    text = (ROOT / "docs" / "GEOMETRY3D_CONTRACT.md").read_text(encoding="utf-8")

    required = [
        "BuildingBox",
        "RoadSegment",
        "Lane",
        "ray-box intersection",
        "LoS/NLoS",
        "blocker_records",
        "Stage 3.1 Implementation Status",
        "stage3_axis_aligned_boxes",
        "not link success",
        "not network delivery",
        "not application consensus",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"GEOMETRY3D_CONTRACT missing Stage 3.1 terms: {missing}"


def test_project_state_records_stage3_1_complete() -> None:
    text = (ROOT / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")

    required = [
        "stage_3_1_geometry_visibility_implementation",
        "stage3_geometry_visibility_gate",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 3.1 state: {missing}"


def test_stage3_1_does_not_add_channel_reward_training_or_consensus_code() -> None:
    banned_terms = [
        "sinr_db",
        "path_loss_db",
        "interference_power",
        "packet_success_probability",
        "consensus_success",
        "consensus_success_probability",
        "def reward",
        "class Reward",
        "train_loop",
        "optimizer",
        "backward(",
        "class Actor(",
        "class Critic(",
        "COMA",
        "MAPPO",
    ]
    scan_roots = [
        ROOT / "src" / "marl_topology" / "geometry3d",
        ROOT / "src" / "marl_topology" / "scenario",
    ]
    offenders: dict[str, list[str]] = {}
    for root in scan_roots:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            hits = [term for term in banned_terms if term in text]
            if hits:
                offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"Stage 3.1 added forbidden lower/upper layer code: {offenders}"
