from pathlib import Path

from marl_topology.channel import evaluate_channel, get_channel_model_fixture
from marl_topology.link import (
    LinkTransmissionConfig,
    LinkTransmissionRecord,
    URLLC_FINITE_BLOCKLENGTH_REGIME_ID,
    evaluate_link_transmission,
    get_link_transmission_fixture,
)


ROOT = Path(__file__).resolve().parents[2]


def test_link_transmission_public_interfaces_exist() -> None:
    channel_fixture = get_channel_model_fixture("free_space_close")
    channel_record = evaluate_channel(
        channel_fixture.scene,
        channel_fixture.tx_id,
        channel_fixture.rx_id,
    )
    link_record = evaluate_link_transmission(channel_record)

    assert LinkTransmissionConfig
    assert LinkTransmissionRecord
    assert get_link_transmission_fixture("payload_small")
    assert link_record.link_transmission_model_id == URLLC_FINITE_BLOCKLENGTH_REGIME_ID
    assert link_record.finite_blocklength_regime_id == URLLC_FINITE_BLOCKLENGTH_REGIME_ID
    assert link_record.channel_model_id == "stage3_channel_v1_fspl_sinr"


def test_stage3_3_link_source_does_not_add_network_reward_training_consensus_or_v5_code() -> None:
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
        "network_delivery",
        "routing",
        "broadcast",
        "marl_topology.metrics",
        "logistic",
        "sigmoid",
    ]
    scan_paths = [
        ROOT / "src" / "marl_topology" / "link" / "transmission.py",
        ROOT / "src" / "marl_topology" / "link" / "transmission_fixtures.py",
    ]
    offenders: dict[str, list[str]] = {}
    for path in scan_paths:
        text = path.read_text(encoding="utf-8")
        hits = [term for term in banned_terms if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"Stage 3.3 link transmission added forbidden code: {offenders}"


def test_link_packet_success_and_latency_are_not_labelled_oracle_or_metric() -> None:
    scan_paths = [
        ROOT / "src" / "marl_topology" / "link" / "transmission.py",
        ROOT / "src" / "marl_topology" / "link" / "transmission_fixtures.py",
    ]
    offenders = {}
    for path in scan_paths:
        text = path.read_text(encoding="utf-8").lower()
        hits = [term for term in ("oracle", "metric") if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"Link transmission fields should not be labelled oracle/metric: {offenders}"
