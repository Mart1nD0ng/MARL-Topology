from marl_topology.channel import evaluate_channel, get_channel_model_fixture
from marl_topology.link import (
    LinkTransmissionConfig,
    URLLC_FINITE_BLOCKLENGTH_REGIME_ID,
    dbm_to_watt,
    deadline_retransmission_probability,
    evaluate_link_transmission,
    finite_blocklength_packet_success_probability,
    get_link_transmission_fixture,
    iter_link_transmission_fixtures,
    required_transmission_time_for_reliability,
    solve_required_transmission_time_for_reliability,
)


def _close_channel_record(tx_power_dbm: float | None = None):
    fixture = get_channel_model_fixture("free_space_close")
    return evaluate_channel(
        fixture.scene,
        fixture.tx_id,
        fixture.rx_id,
        tx_power_dbm=tx_power_dbm,
    )


def test_link_transmission_fixtures_are_evaluable() -> None:
    for fixture in iter_link_transmission_fixtures():
        record = fixture.evaluate()

        assert record.link_transmission_model_id == URLLC_FINITE_BLOCKLENGTH_REGIME_ID
        assert record.p2p_latency_s >= 0.0
        assert record.p2p_energy_j >= 0.0
        assert 0.0 <= record.packet_error_probability <= 1.0
        assert 0.0 <= record.packet_success_probability <= 1.0
        assert 0.0 <= record.deadline_delivery_probability <= 1.0
        assert record.finite_blocklength_regime_id == URLLC_FINITE_BLOCKLENGTH_REGIME_ID
        assert 0.0 <= record.success_probability_at_required_time <= 1.0
        assert record.required_time_search_max_s > 0.0


def test_larger_payload_increases_transmission_delay_and_energy() -> None:
    channel_record = _close_channel_record(tx_power_dbm=-10.0)

    small = evaluate_link_transmission(
        channel_record,
        LinkTransmissionConfig(payload_bits=4_000, fixed_transmission_time_s=0.0005),
    )
    large = evaluate_link_transmission(
        channel_record,
        LinkTransmissionConfig(payload_bits=40_000, fixed_transmission_time_s=0.0005),
    )

    assert large.required_transmission_time_s > small.required_transmission_time_s
    assert large.packet_success_probability < small.packet_success_probability


def test_larger_bandwidth_decreases_transmission_delay() -> None:
    channel_record = _close_channel_record(tx_power_dbm=-10.0)

    narrow = evaluate_link_transmission(
        channel_record,
        LinkTransmissionConfig(bandwidth_hz=5e6, fixed_transmission_time_s=0.0005),
    )
    wide = evaluate_link_transmission(
        channel_record,
        LinkTransmissionConfig(bandwidth_hz=20e6, fixed_transmission_time_s=0.0005),
    )

    assert wide.packet_success_probability > narrow.packet_success_probability
    assert wide.required_transmission_time_s < narrow.required_transmission_time_s


def test_larger_tx_power_increases_tx_energy_for_fixed_payload() -> None:
    low_power = evaluate_link_transmission(_close_channel_record(tx_power_dbm=10.0))
    high_power = evaluate_link_transmission(_close_channel_record(tx_power_dbm=25.0))

    assert high_power.tx_power_w > low_power.tx_power_w
    assert high_power.tx_energy_j > low_power.tx_energy_j


def test_larger_distance_increases_propagation_delay() -> None:
    close_fixture = get_channel_model_fixture("free_space_close")
    far_fixture = get_channel_model_fixture("free_space_far")
    close_channel = evaluate_channel(close_fixture.scene, close_fixture.tx_id, close_fixture.rx_id)
    far_channel = evaluate_channel(far_fixture.scene, far_fixture.tx_id, far_fixture.rx_id)

    close_link = evaluate_link_transmission(close_channel)
    far_link = evaluate_link_transmission(far_channel)

    assert far_link.distance_3d_m > close_link.distance_3d_m
    assert far_link.propagation_delay_s > close_link.propagation_delay_s


def test_latency_and_energy_components_are_nonnegative() -> None:
    record = evaluate_link_transmission(_close_channel_record())

    assert record.propagation_delay_s >= 0.0
    assert record.transmission_delay_s >= 0.0
    assert record.processing_delay_s >= 0.0
    assert record.queueing_delay_s >= 0.0
    assert record.p2p_latency_s >= 0.0
    assert record.tx_energy_j >= 0.0
    assert record.rx_energy_j >= 0.0
    assert record.processing_energy_j >= 0.0
    assert record.p2p_energy_j >= 0.0


def test_unselected_or_inactive_edge_consumes_no_transmission_energy() -> None:
    channel_record = _close_channel_record()
    unselected = evaluate_link_transmission(channel_record, selected=False)
    inactive = evaluate_link_transmission(channel_record, active=False)
    fixture_record = get_link_transmission_fixture("unselected_edge").evaluate()

    for record in (unselected, inactive, fixture_record):
        assert record.transmission_attempted is False
        assert record.p2p_latency_s == 0.0
        assert record.p2p_energy_j == 0.0
        assert record.tx_energy_j == 0.0
        assert record.rx_energy_j == 0.0
        assert record.processing_energy_j == 0.0


def test_dbm_to_watt_conversion_is_monotonic() -> None:
    assert dbm_to_watt(20.0) > dbm_to_watt(10.0) > 0.0


def test_finite_blocklength_success_increases_with_sinr_bandwidth_and_duration() -> None:
    low_sinr = finite_blocklength_packet_success_probability(
        sinr_db=-5.0,
        bandwidth_hz=10e6,
        transmission_time_s=0.002,
        payload_bits=12_000,
    )
    high_sinr = finite_blocklength_packet_success_probability(
        sinr_db=20.0,
        bandwidth_hz=10e6,
        transmission_time_s=0.002,
        payload_bits=12_000,
    )
    narrow = finite_blocklength_packet_success_probability(
        sinr_db=10.0,
        bandwidth_hz=5e6,
        transmission_time_s=0.0005,
        payload_bits=12_000,
    )
    wide = finite_blocklength_packet_success_probability(
        sinr_db=10.0,
        bandwidth_hz=20e6,
        transmission_time_s=0.0005,
        payload_bits=12_000,
    )
    short = finite_blocklength_packet_success_probability(
        sinr_db=5.0,
        bandwidth_hz=10e6,
        transmission_time_s=0.0005,
        payload_bits=12_000,
    )
    long = finite_blocklength_packet_success_probability(
        sinr_db=5.0,
        bandwidth_hz=10e6,
        transmission_time_s=0.0008,
        payload_bits=12_000,
    )

    assert 0.0 <= low_sinr < high_sinr <= 1.0
    assert 0.0 <= narrow < wide <= 1.0
    assert 0.0 <= short < long <= 1.0


def test_inverse_reliability_time_increases_with_payload_and_decreases_with_sinr() -> None:
    small_payload = required_transmission_time_for_reliability(
        sinr_db=8.0,
        bandwidth_hz=10e6,
        payload_bits=4_000,
        target_reliability=0.99,
    )
    large_payload = required_transmission_time_for_reliability(
        sinr_db=8.0,
        bandwidth_hz=10e6,
        payload_bits=40_000,
        target_reliability=0.99,
    )
    low_sinr = required_transmission_time_for_reliability(
        sinr_db=2.0,
        bandwidth_hz=10e6,
        payload_bits=12_000,
        target_reliability=0.99,
    )
    high_sinr = required_transmission_time_for_reliability(
        sinr_db=20.0,
        bandwidth_hz=10e6,
        payload_bits=12_000,
        target_reliability=0.99,
    )

    assert large_payload > small_payload
    assert low_sinr > high_sinr


def test_unreachable_inverse_reliability_target_is_explicitly_capped() -> None:
    result = solve_required_transmission_time_for_reliability(
        sinr_db=-40.0,
        bandwidth_hz=1e3,
        payload_bits=10_000_000,
        target_reliability=0.999999,
        max_time_s=0.001,
    )

    assert result.required_transmission_time_s == 0.001
    assert result.required_reliability_met is False
    assert result.required_transmission_time_capped is True
    assert result.success_probability_at_required_time < 0.999999
    assert result.required_time_search_max_s == 0.001


def test_unreachable_inverse_reliability_record_exposes_diagnostics() -> None:
    record = evaluate_link_transmission(
        _close_channel_record(tx_power_dbm=-80.0),
        LinkTransmissionConfig(
            payload_bits=10_000_000,
            bandwidth_hz=1e3,
            target_reliability=0.999999,
            use_inverse_reliability=True,
            max_required_transmission_time_s=0.001,
        ),
    )

    assert record.required_reliability_met is False
    assert record.required_transmission_time_capped is True
    assert record.required_transmission_time_s == 0.001
    assert record.success_probability_at_required_time < 0.999999


def test_deadline_retransmission_probability_and_expected_energy_are_coupled() -> None:
    one_attempt = evaluate_link_transmission(
        _close_channel_record(),
        LinkTransmissionConfig(fixed_transmission_time_s=0.001, deadline_s=0.0015),
    )
    three_attempts = evaluate_link_transmission(
        _close_channel_record(),
        LinkTransmissionConfig(fixed_transmission_time_s=0.001, deadline_s=0.006),
    )

    assert three_attempts.max_attempts_within_deadline > one_attempt.max_attempts_within_deadline
    assert three_attempts.deadline_delivery_probability >= one_attempt.deadline_delivery_probability
    assert three_attempts.expected_attempts >= one_attempt.expected_attempts
    assert three_attempts.expected_energy_j >= one_attempt.expected_energy_j
    assert three_attempts.p2p_latency_s == three_attempts.expected_latency_s
    assert three_attempts.p2p_energy_j == three_attempts.expected_energy_j


def test_deadline_retransmission_edge_cases_are_finite() -> None:
    zero_attempts = deadline_retransmission_probability(
        packet_success_probability=0.5,
        attempt_duration_s=0.01,
        deadline_s=0.0,
    )
    zero_success = deadline_retransmission_probability(
        packet_success_probability=0.0,
        attempt_duration_s=0.01,
        deadline_s=0.03,
    )

    assert zero_attempts == (0, 0.0, 0.0)
    assert zero_success == (2, 0.0, 2.0)
