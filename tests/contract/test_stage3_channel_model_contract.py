from pathlib import Path

from marl_topology.channel import (
    ActiveTransmission,
    ChannelModelConfig,
    ChannelRecord,
    STAGE3_CHANNEL_REGIME_ID,
    evaluate_channel,
    get_channel_model_fixture,
)


ROOT = Path(__file__).resolve().parents[2]


def test_channel_model_public_interfaces_exist() -> None:
    fixture = get_channel_model_fixture("free_space_close")
    record = evaluate_channel(fixture.scene, fixture.tx_id, fixture.rx_id)

    assert ActiveTransmission
    assert ChannelModelConfig
    assert ChannelRecord
    assert record.channel_model_id == STAGE3_CHANNEL_REGIME_ID
    assert record.visibility_regime == "stage3_axis_aligned_boxes"
    assert record.resource_id == "resource_0"


def test_stage3_2_channel_source_does_not_add_reward_training_consensus_or_v5_code() -> None:
    banned_terms = [
        "consensus_success",
        "consensus_success_probability",
        "P_eff",
        "quorum",
        "PBFT",
        "def reward",
        "class Reward",
        "train_loop",
        "optimizer",
        "backward(",
        "class Actor(",
        "class Critic(",
        "COMA",
        "MAPPO",
        "D:\\PhD_works\\v5",
        "phase",
        "marl_topology.metrics",
    ]
    offenders: dict[str, list[str]] = {}
    for path in (ROOT / "src" / "marl_topology" / "channel").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        hits = [term for term in banned_terms if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"Stage 3.2 channel added forbidden code: {offenders}"


def test_channel_packet_success_is_not_labelled_oracle_or_metric() -> None:
    text_by_file = {
        str(path.relative_to(ROOT)): path.read_text(encoding="utf-8")
        for path in (ROOT / "src" / "marl_topology" / "channel").rglob("*.py")
    }
    offenders = {
        path: ["oracle"]
        for path, text in text_by_file.items()
        if "oracle" in text.lower()
    }

    assert not offenders, f"Channel layer should not label packet success as oracle: {offenders}"
