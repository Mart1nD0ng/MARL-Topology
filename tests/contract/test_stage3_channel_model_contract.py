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


def test_channel_model_contract_records_stage3_2_implementation_status() -> None:
    text = (ROOT / "docs" / "CHANNEL_MODEL_CONTRACT.md").read_text(encoding="utf-8")

    required = [
        "Stage 3.2 Implementation Status",
        "stage3_channel_v1_fspl_sinr",
        "ChannelRecord",
        "ActiveTransmission",
        "path_loss_db",
        "noise_power_dbm",
        "interference_power_dbm",
        "sinr_db",
        "packet success is not computed by the channel layer",
        "not consensus success",
        "not PBFT reliability",
        "not reward",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"CHANNEL_MODEL_CONTRACT missing Stage 3.2 terms: {missing}"


def test_project_state_records_stage3_2_complete() -> None:
    text = (ROOT / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")

    required = [
        "stage_3_2_channel_model_v1",
        "stage3_channel_model_gate",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 3.2 state: {missing}"


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
