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


def test_stage3_6_docs_record_urlcc_finite_blocklength_regime() -> None:
    channel = _read_doc("CHANNEL_MODEL_CONTRACT.md")
    link = _read_doc("LINK_TRANSMISSION_CONTRACT.md")
    coupling = _read_doc("STAGE3_LATENCY_ENERGY_RELIABILITY_COUPLING_REVIEW.md")

    channel_required = [
        "stage3_channel_v1_fspl_sinr",
        "Packet success is not computed by the channel layer",
        "removed from the active code path",
    ]
    link_required = [
        "urlcc_finite_blocklength_v1",
        "packet_error_probability",
        "required_transmission_time_s",
        "deadline_delivery_probability",
        "expected_attempts",
        "expected_latency_s",
        "expected_energy_j",
        "K = floor(deadline_s / attempt_duration_s)",
    ]
    coupling_required = [
        "urlcc_finite_blocklength_v1",
        "packet_success_probability = 1 - epsilon",
        "deadline_delivery_probability",
        "expected_energy_j = expected_attempts * attempt_energy_j",
        "Communication delivery is not `consensus_success_probability`",
    ]

    missing_channel = [item for item in channel_required if item not in channel]
    missing_link = [item for item in link_required if item not in link]
    missing_coupling = [item for item in coupling_required if item not in coupling]

    assert not missing_channel, f"CHANNEL_MODEL_CONTRACT missing Stage 3.6 terms: {missing_channel}"
    assert not missing_link, f"LINK_TRANSMISSION_CONTRACT missing Stage 3.6 terms: {missing_link}"
    assert not missing_coupling, f"coupling review missing Stage 3.6 terms: {missing_coupling}"


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


def test_stage3_6_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage3_6_urlcc_finite_blocklength_link_reliability.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage3_6_urlcc_finite_blocklength_link_reliability"
    for field in [
        "required_artifacts",
        "expected_evidence",
        "negative_checks",
        "relevant_lessons",
        "required_tests",
        "forbidden_v5_inheritance",
        "expected_outputs",
    ]:
        assert field in task
        assert task[field]
    negative_text = " ".join(task["negative_checks"])
    assert "old SINR-only success surrogate" in negative_text
    assert "reward" in negative_text
    assert "v5" in negative_text


def test_old_active_success_surrogate_removed_from_active_source_and_docs() -> None:
    scan_paths = [
        ROOT / "src" / "marl_topology" / "channel" / "model.py",
        ROOT / "src" / "marl_topology" / "channel" / "__init__.py",
        ROOT / "src" / "marl_topology" / "link" / "transmission.py",
        ROOT / "docs" / "CHANNEL_MODEL_CONTRACT.md",
        ROOT / "docs" / "LINK_TRANSMISSION_CONTRACT.md",
        ROOT / "docs" / "PHYSICS_CONTRACT.md",
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
