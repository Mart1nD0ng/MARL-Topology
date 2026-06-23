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
