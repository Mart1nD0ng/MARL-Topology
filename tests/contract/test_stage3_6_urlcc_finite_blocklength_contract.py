from pathlib import Path

import yaml

from marl_topology.channel import ChannelRecord, STAGE3_CHANNEL_REGIME_ID
from marl_topology.link import (
    LinkTransmissionRecord,
    URLLC_FINITE_BLOCKLENGTH_REGIME_ID,
)


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage3_6_public_records_expose_new_regime_fields() -> None:
    assert ChannelRecord
    assert LinkTransmissionRecord
    assert STAGE3_CHANNEL_REGIME_ID == "stage3_channel_v1_fspl_sinr"
    assert URLLC_FINITE_BLOCKLENGTH_REGIME_ID == "urlcc_finite_blocklength_v1"
    assert "packet_success_probability" not in ChannelRecord.__annotations__
    for field_name in [
        "packet_error_probability",
        "packet_success_probability",
        "target_reliability",
        "required_transmission_time_s",
        "attempt_duration_s",
        "max_attempts_within_deadline",
        "deadline_delivery_probability",
        "expected_attempts",
        "expected_latency_s",
        "expected_energy_j",
        "finite_blocklength_regime_id",
    ]:
        assert field_name in LinkTransmissionRecord.__annotations__


def test_old_active_success_surrogate_removed_from_active_source_and_docs() -> None:
    # The CHANNEL_MODEL / LINK_TRANSMISSION / PHYSICS contract docs were retired with the
    # multi-stage process lineage (2026-06-23); the durable invariant is that the old success
    # surrogate stays out of the active SOURCE.
    scan_paths = [
        ROOT / "src" / "marl_topology" / "channel" / "model.py",
        ROOT / "src" / "marl_topology" / "channel" / "__init__.py",
        ROOT / "src" / "marl_topology" / "link" / "transmission.py",
    ]
    banned_terms = [
        "stage3_channel_v1_fspl_logistic",
        "packet_success_probability_from_sinr",
        "sigmoid",
    ]
    offenders: dict[str, list[str]] = {}
    for path in scan_paths:
        text = path.read_text(encoding="utf-8")
        hits = [term for term in banned_terms if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"old active success surrogate remains: {offenders}"


def test_stage3_6_source_does_not_add_reward_training_consensus_or_v5_code() -> None:
    banned_terms = [
        "P_eff",
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
        "marl_topology.metrics",
    ]
    scan_paths = [
        ROOT / "src" / "marl_topology" / "channel" / "model.py",
        ROOT / "src" / "marl_topology" / "link" / "transmission.py",
    ]
    offenders: dict[str, list[str]] = {}
    for path in scan_paths:
        text = path.read_text(encoding="utf-8")
        hits = [term for term in banned_terms if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"Stage 3.6 added forbidden code: {offenders}"
