from pathlib import Path

from marl_topology.network import (
    NetworkCommunicationConfig,
    NetworkCommunicationRecord,
    NetworkHopRecord,
    NetworkTransmissionSpec,
    get_network_communication_fixture,
)


ROOT = Path(__file__).resolve().parents[2]


def test_network_layer_public_interfaces_exist() -> None:
    fixture = get_network_communication_fixture("multi_hop_delivery")
    record = fixture.evaluate()

    assert NetworkCommunicationConfig
    assert NetworkCommunicationRecord
    assert NetworkHopRecord
    assert NetworkTransmissionSpec
    assert record.network_model_id == "stage3_network_communication_v1"
    assert record.network_regime == "stage3_network_communication_v1"


def test_network_layer_contract_records_stage3_4_implementation_status() -> None:
    text = (ROOT / "docs" / "NETWORK_LAYER_CONTRACT.md").read_text(encoding="utf-8")

    required = [
        "Stage 3.4 Implementation Status",
        "stage3_network_communication_v1",
        "NetworkCommunicationConfig",
        "NetworkCommunicationRecord",
        "NetworkHopRecord",
        "NetworkTransmissionSpec",
        "evaluate_network_communication",
        "network_delivery_probability",
        "network_latency_s",
        "network_energy_j",
        "not PBFT quorum success",
        "not consensus reliability",
        "not reward",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"NETWORK_LAYER_CONTRACT missing Stage 3.4 terms: {missing}"


def test_project_state_records_stage3_4_complete() -> None:
    text = (ROOT / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")

    required = [
        "stage_3_4_network_layer_communication_v1",
        "stage3_network_layer_gate",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 3.4 state: {missing}"


def test_stage3_4_network_source_does_not_add_reward_training_consensus_or_v5_code() -> None:
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
    for path in (ROOT / "src" / "marl_topology" / "network").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        hits = [term for term in banned_terms if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"Stage 3.4 network added forbidden code: {offenders}"


def test_network_delivery_is_not_labelled_as_metric_and_oracle_flag_is_false() -> None:
    offenders = {}
    for path in (ROOT / "src" / "marl_topology" / "network").rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        hits = [term for term in ("metric",) if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    record = get_network_communication_fixture("multi_hop_delivery").evaluate()

    assert not offenders, f"Network fields should not be labelled metric: {offenders}"
    assert record.is_oracle is False
