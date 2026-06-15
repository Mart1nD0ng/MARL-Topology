"""Stage 3 point-to-point URLLC finite-blocklength transmission records."""

from __future__ import annotations

from dataclasses import dataclass
from math import erfc, isfinite, log2, sqrt

from marl_topology.channel import ChannelRecord, dbm_to_mw


SPEED_OF_LIGHT_MPS = 299_792_458.0
MILLIWATT_TO_WATT = 0.001
URLLC_FINITE_BLOCKLENGTH_REGIME_ID = "urlcc_finite_blocklength_v1"
LOG2_E = log2(2.718281828459045)


@dataclass(frozen=True, slots=True)
class LinkTransmissionConfig:
    link_transmission_model_id: str = URLLC_FINITE_BLOCKLENGTH_REGIME_ID
    payload_bits: int = 12_000
    spectral_efficiency_bps_per_hz: float = 1.0
    processing_delay_s: float = 0.0001
    queueing_delay_s: float = 0.0
    rx_circuit_power_w: float = 0.2
    processing_power_w: float = 0.1
    propagation_speed_mps: float = SPEED_OF_LIGHT_MPS
    bandwidth_hz: float | None = None
    fixed_transmission_time_s: float | None = None
    target_reliability: float | None = 0.99
    use_inverse_reliability: bool = False
    deadline_s: float | None = None
    use_normal_approximation_correction: bool = True
    max_required_transmission_time_s: float = 10.0

    def __post_init__(self) -> None:
        if not self.link_transmission_model_id:
            raise ValueError("link_transmission_model_id must be non-empty")
        if self.payload_bits < 0:
            raise ValueError("payload_bits must be nonnegative")
        if self.spectral_efficiency_bps_per_hz <= 0:
            raise ValueError("spectral_efficiency_bps_per_hz must be positive")
        if self.processing_delay_s < 0:
            raise ValueError("processing_delay_s must be nonnegative")
        if self.queueing_delay_s < 0:
            raise ValueError("queueing_delay_s must be nonnegative")
        if self.rx_circuit_power_w < 0:
            raise ValueError("rx_circuit_power_w must be nonnegative")
        if self.processing_power_w < 0:
            raise ValueError("processing_power_w must be nonnegative")
        if self.propagation_speed_mps <= 0:
            raise ValueError("propagation_speed_mps must be positive")
        if self.bandwidth_hz is not None and self.bandwidth_hz <= 0:
            raise ValueError("bandwidth_hz override must be positive")
        if self.fixed_transmission_time_s is not None and self.fixed_transmission_time_s <= 0:
            raise ValueError("fixed_transmission_time_s must be positive when provided")
        if self.target_reliability is not None and not 0.0 < self.target_reliability < 1.0:
            raise ValueError("target_reliability must be in (0, 1) when provided")
        if self.deadline_s is not None and self.deadline_s < 0:
            raise ValueError("deadline_s must be nonnegative when provided")
        if self.max_required_transmission_time_s <= 0:
            raise ValueError("max_required_transmission_time_s must be positive")


@dataclass(frozen=True, slots=True)
class RequiredTransmissionTimeResult:
    required_transmission_time_s: float
    required_reliability_met: bool
    required_transmission_time_capped: bool
    success_probability_at_required_time: float
    required_time_search_max_s: float

    def __post_init__(self) -> None:
        if self.required_transmission_time_s < 0.0:
            raise ValueError("required_transmission_time_s must be nonnegative")
        if self.required_time_search_max_s <= 0.0:
            raise ValueError("required_time_search_max_s must be positive")
        if not 0.0 <= self.success_probability_at_required_time <= 1.0:
            raise ValueError("success_probability_at_required_time must be in [0, 1]")
        if self.required_transmission_time_capped and self.required_reliability_met:
            raise ValueError("capped required time cannot meet the target")


@dataclass(frozen=True, slots=True)
class LinkTransmissionRecord:
    scenario_id: str
    tx_id: str
    rx_id: str
    resource_id: str
    channel_model_id: str
    link_transmission_model_id: str
    distance_3d_m: float
    payload_bits: int
    bandwidth_hz: float
    spectral_efficiency_bps_per_hz: float
    effective_rate_bps: float
    selected: bool
    active: bool
    transmission_attempted: bool
    propagation_delay_s: float
    transmission_delay_s: float
    processing_delay_s: float
    queueing_delay_s: float
    p2p_latency_s: float
    tx_power_w: float
    rx_circuit_power_w: float
    processing_power_w: float
    tx_energy_j: float
    rx_energy_j: float
    processing_energy_j: float
    p2p_energy_j: float
    packet_error_probability: float
    packet_success_probability: float
    target_reliability: float | None
    required_transmission_time_s: float
    required_reliability_met: bool
    required_transmission_time_capped: bool
    success_probability_at_required_time: float
    required_time_search_max_s: float
    attempt_duration_s: float
    max_attempts_within_deadline: int
    deadline_delivery_probability: float
    expected_attempts: float
    expected_latency_s: float
    expected_energy_j: float
    finite_blocklength_regime_id: str
    link_transmission_regime: str = "stage3_link_transmission_v1"

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must be non-empty")
        if not self.tx_id or not self.rx_id:
            raise ValueError("tx_id and rx_id must be non-empty")
        if self.tx_id == self.rx_id:
            raise ValueError("tx_id and rx_id must differ")
        if not self.resource_id:
            raise ValueError("resource_id must be non-empty")
        if not self.channel_model_id or not self.link_transmission_model_id:
            raise ValueError("model ids must be non-empty")
        if self.distance_3d_m < 0:
            raise ValueError("distance_3d_m must be nonnegative")
        if self.payload_bits < 0:
            raise ValueError("payload_bits must be nonnegative")
        if self.bandwidth_hz <= 0 or self.effective_rate_bps <= 0:
            raise ValueError("bandwidth_hz and effective_rate_bps must be positive")
        if self.spectral_efficiency_bps_per_hz <= 0:
            raise ValueError("spectral_efficiency_bps_per_hz must be positive")
        delay_values = (
            self.propagation_delay_s,
            self.transmission_delay_s,
            self.processing_delay_s,
            self.queueing_delay_s,
            self.p2p_latency_s,
        )
        if any(value < 0 for value in delay_values):
            raise ValueError("latency components must be nonnegative")
        power_values = (self.tx_power_w, self.rx_circuit_power_w, self.processing_power_w)
        if any(value < 0 for value in power_values):
            raise ValueError("power components must be nonnegative")
        energy_values = (
            self.tx_energy_j,
            self.rx_energy_j,
            self.processing_energy_j,
            self.p2p_energy_j,
        )
        if any(value < 0 for value in energy_values):
            raise ValueError("energy components must be nonnegative")
        for probability_name, probability in (
            ("packet_error_probability", self.packet_error_probability),
            ("packet_success_probability", self.packet_success_probability),
            ("deadline_delivery_probability", self.deadline_delivery_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{probability_name} must be in [0, 1]")
        if self.target_reliability is not None and not 0.0 < self.target_reliability < 1.0:
            raise ValueError("target_reliability must be in (0, 1) when provided")
        if self.required_transmission_time_s < 0 or self.attempt_duration_s < 0:
            raise ValueError("finite-blocklength durations must be nonnegative")
        if not 0.0 <= self.success_probability_at_required_time <= 1.0:
            raise ValueError("success_probability_at_required_time must be in [0, 1]")
        if self.required_time_search_max_s <= 0.0:
            raise ValueError("required_time_search_max_s must be positive")
        if self.required_transmission_time_capped and self.required_reliability_met:
            raise ValueError("capped required time cannot meet the target")
        if self.max_attempts_within_deadline < 0:
            raise ValueError("max_attempts_within_deadline must be nonnegative")
        if self.expected_attempts < 0:
            raise ValueError("expected_attempts must be nonnegative")
        if self.expected_latency_s < 0 or self.expected_energy_j < 0:
            raise ValueError("expected latency and energy must be nonnegative")
        if not self.finite_blocklength_regime_id:
            raise ValueError("finite_blocklength_regime_id must be non-empty")
        if not self.transmission_attempted and (
            self.p2p_latency_s != 0.0 or self.p2p_energy_j != 0.0
        ):
            raise ValueError("inactive or unselected transmissions must have zero latency/energy")


def dbm_to_watt(power_dbm: float) -> float:
    return dbm_to_mw(power_dbm) * MILLIWATT_TO_WATT


def sinr_db_to_linear(sinr_db: float) -> float:
    if not isfinite(sinr_db):
        raise ValueError("sinr_db must be finite")
    return 10.0 ** (sinr_db / 10.0)


def finite_blocklength_packet_error_probability(
    *,
    sinr_db: float,
    bandwidth_hz: float,
    transmission_time_s: float,
    payload_bits: int,
    use_normal_approximation_correction: bool = True,
) -> float:
    """Return a finite-blocklength normal-approximation packet error rate."""

    if bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be positive")
    if transmission_time_s <= 0:
        raise ValueError("transmission_time_s must be positive")
    if payload_bits < 0:
        raise ValueError("payload_bits must be nonnegative")
    gamma = sinr_db_to_linear(sinr_db)
    n_symbols = bandwidth_hz * transmission_time_s
    if n_symbols <= 0.0 or not isfinite(n_symbols):
        return 1.0
    capacity_bits_per_symbol = log2(1.0 + gamma)
    dispersion = (1.0 - (1.0 + gamma) ** -2.0) * (LOG2_E**2)
    if dispersion <= 0.0:
        return 0.0 if payload_bits <= 0 else 1.0
    numerator = n_symbols * capacity_bits_per_symbol - float(payload_bits)
    if use_normal_approximation_correction:
        numerator += 0.5 * log2(n_symbols)
    denominator = sqrt(n_symbols * dispersion)
    if denominator <= 0.0 or not isfinite(denominator):
        return 1.0
    q_argument = numerator / denominator
    return _clip_probability(0.5 * erfc(q_argument / sqrt(2.0)))


def finite_blocklength_packet_success_probability(
    *,
    sinr_db: float,
    bandwidth_hz: float,
    transmission_time_s: float,
    payload_bits: int,
    use_normal_approximation_correction: bool = True,
) -> float:
    error_probability = finite_blocklength_packet_error_probability(
        sinr_db=sinr_db,
        bandwidth_hz=bandwidth_hz,
        transmission_time_s=transmission_time_s,
        payload_bits=payload_bits,
        use_normal_approximation_correction=use_normal_approximation_correction,
    )
    return _clip_probability(1.0 - error_probability)


def required_transmission_time_for_reliability(
    *,
    sinr_db: float,
    bandwidth_hz: float,
    payload_bits: int,
    target_reliability: float,
    use_normal_approximation_correction: bool = True,
    max_time_s: float = 10.0,
) -> float:
    """Solve the minimal fixed transmission time needed for target reliability."""

    return solve_required_transmission_time_for_reliability(
        sinr_db=sinr_db,
        bandwidth_hz=bandwidth_hz,
        payload_bits=payload_bits,
        target_reliability=target_reliability,
        use_normal_approximation_correction=use_normal_approximation_correction,
        max_time_s=max_time_s,
    ).required_transmission_time_s


def solve_required_transmission_time_for_reliability(
    *,
    sinr_db: float,
    bandwidth_hz: float,
    payload_bits: int,
    target_reliability: float,
    use_normal_approximation_correction: bool = True,
    max_time_s: float = 10.0,
) -> RequiredTransmissionTimeResult:
    """Solve required time and expose whether the search cap was binding."""

    if not 0.0 < target_reliability < 1.0:
        raise ValueError("target_reliability must be in (0, 1)")
    if bandwidth_hz <= 0.0:
        raise ValueError("bandwidth_hz must be positive")
    if max_time_s <= 0.0:
        raise ValueError("max_time_s must be positive")
    if payload_bits == 0:
        return RequiredTransmissionTimeResult(
            required_transmission_time_s=0.0,
            required_reliability_met=True,
            required_transmission_time_capped=False,
            success_probability_at_required_time=1.0,
            required_time_search_max_s=max_time_s,
        )
    lower = 1.0 / bandwidth_hz
    upper = lower
    while upper < max_time_s:
        probability = finite_blocklength_packet_success_probability(
            sinr_db=sinr_db,
            bandwidth_hz=bandwidth_hz,
            transmission_time_s=upper,
            payload_bits=payload_bits,
            use_normal_approximation_correction=use_normal_approximation_correction,
        )
        if probability >= target_reliability:
            break
        upper *= 2.0
    else:
        upper = max_time_s
    probability_at_upper = finite_blocklength_packet_success_probability(
        sinr_db=sinr_db,
        bandwidth_hz=bandwidth_hz,
        transmission_time_s=upper,
        payload_bits=payload_bits,
        use_normal_approximation_correction=use_normal_approximation_correction,
    )
    if probability_at_upper < target_reliability:
        return RequiredTransmissionTimeResult(
            required_transmission_time_s=upper,
            required_reliability_met=False,
            required_transmission_time_capped=True,
            success_probability_at_required_time=probability_at_upper,
            required_time_search_max_s=max_time_s,
        )
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        probability = finite_blocklength_packet_success_probability(
            sinr_db=sinr_db,
            bandwidth_hz=bandwidth_hz,
            transmission_time_s=midpoint,
            payload_bits=payload_bits,
            use_normal_approximation_correction=use_normal_approximation_correction,
        )
        if probability >= target_reliability:
            upper = midpoint
        else:
            lower = midpoint
    probability_at_required_time = finite_blocklength_packet_success_probability(
        sinr_db=sinr_db,
        bandwidth_hz=bandwidth_hz,
        transmission_time_s=upper,
        payload_bits=payload_bits,
        use_normal_approximation_correction=use_normal_approximation_correction,
    )
    return RequiredTransmissionTimeResult(
        required_transmission_time_s=upper,
        required_reliability_met=probability_at_required_time >= target_reliability,
        required_transmission_time_capped=False,
        success_probability_at_required_time=probability_at_required_time,
        required_time_search_max_s=max_time_s,
    )


def deadline_retransmission_probability(
    *,
    packet_success_probability: float,
    attempt_duration_s: float,
    deadline_s: float,
) -> tuple[int, float, float]:
    """Return max attempts, deadline delivery probability, and expected attempts."""

    if not 0.0 <= packet_success_probability <= 1.0:
        raise ValueError("packet_success_probability must be in [0, 1]")
    if attempt_duration_s <= 0.0:
        raise ValueError("attempt_duration_s must be positive")
    if deadline_s < 0.0:
        raise ValueError("deadline_s must be nonnegative")
    max_attempts = int(deadline_s // attempt_duration_s)
    if max_attempts == 0:
        return 0, 0.0, 0.0
    p_attempt = packet_success_probability
    failure_probability = 1.0 - p_attempt
    delivery_probability = _clip_probability(1.0 - failure_probability**max_attempts)
    if p_attempt == 0.0:
        expected_attempts = float(max_attempts)
    else:
        expected_attempts = delivery_probability / p_attempt
    return max_attempts, delivery_probability, expected_attempts


def evaluate_link_transmission(
    channel_record: ChannelRecord,
    config: LinkTransmissionConfig | None = None,
    *,
    selected: bool = True,
    active: bool = True,
) -> LinkTransmissionRecord:
    if config is None:
        config = LinkTransmissionConfig()

    bandwidth_hz = channel_record.bandwidth_hz if config.bandwidth_hz is None else config.bandwidth_hz
    tx_power_w = dbm_to_watt(channel_record.tx_power_dbm)
    transmission_attempted = selected and active and config.payload_bits > 0

    if transmission_attempted:
        propagation_delay_s = channel_record.distance_3d_m / config.propagation_speed_mps
        target_reliability = config.target_reliability
        required_time_result = (
            solve_required_transmission_time_for_reliability(
                sinr_db=channel_record.sinr_db,
                bandwidth_hz=bandwidth_hz,
                payload_bits=config.payload_bits,
                target_reliability=target_reliability,
                use_normal_approximation_correction=config.use_normal_approximation_correction,
                max_time_s=config.max_required_transmission_time_s,
            )
            if target_reliability is not None
            else RequiredTransmissionTimeResult(
                required_transmission_time_s=0.0,
                required_reliability_met=False,
                required_transmission_time_capped=False,
                success_probability_at_required_time=0.0,
                required_time_search_max_s=config.max_required_transmission_time_s,
            )
        )
        required_transmission_time_s = required_time_result.required_transmission_time_s
        if config.use_inverse_reliability:
            if target_reliability is None:
                raise ValueError("target_reliability is required in inverse reliability mode")
            transmission_delay_s = required_transmission_time_s
        elif config.fixed_transmission_time_s is not None:
            transmission_delay_s = config.fixed_transmission_time_s
        else:
            transmission_delay_s = config.payload_bits / (
                bandwidth_hz * config.spectral_efficiency_bps_per_hz
            )
        effective_rate_bps = config.payload_bits / transmission_delay_s
        processing_delay_s = config.processing_delay_s
        queueing_delay_s = config.queueing_delay_s
        attempt_duration_s = (
            propagation_delay_s
            + transmission_delay_s
            + processing_delay_s
            + queueing_delay_s
        )
        packet_error_probability = finite_blocklength_packet_error_probability(
            sinr_db=channel_record.sinr_db,
            bandwidth_hz=bandwidth_hz,
            transmission_time_s=transmission_delay_s,
            payload_bits=config.payload_bits,
            use_normal_approximation_correction=config.use_normal_approximation_correction,
        )
        packet_success_probability = _clip_probability(1.0 - packet_error_probability)
        if config.deadline_s is None:
            max_attempts = 1
            deadline_delivery_probability = packet_success_probability
            expected_attempts = 1.0
        else:
            max_attempts, deadline_delivery_probability, expected_attempts = (
                deadline_retransmission_probability(
                    packet_success_probability=packet_success_probability,
                    attempt_duration_s=attempt_duration_s,
                    deadline_s=config.deadline_s,
                )
            )
        tx_energy_per_attempt_j = tx_power_w * transmission_delay_s
        rx_energy_per_attempt_j = config.rx_circuit_power_w * transmission_delay_s
        processing_energy_j = config.processing_power_w * processing_delay_s
        attempt_energy_j = (
            tx_energy_per_attempt_j + rx_energy_per_attempt_j + processing_energy_j
        )
        expected_latency_s = expected_attempts * attempt_duration_s
        expected_energy_j = expected_attempts * attempt_energy_j
        p2p_latency_s = expected_latency_s
        tx_energy_j = expected_attempts * tx_energy_per_attempt_j
        rx_energy_j = expected_attempts * rx_energy_per_attempt_j
        processing_energy_j = expected_attempts * processing_energy_j
        p2p_energy_j = expected_energy_j
    else:
        effective_rate_bps = bandwidth_hz * config.spectral_efficiency_bps_per_hz
        propagation_delay_s = 0.0
        transmission_delay_s = 0.0
        processing_delay_s = 0.0
        queueing_delay_s = 0.0
        attempt_duration_s = 0.0
        p2p_latency_s = 0.0
        tx_energy_j = 0.0
        rx_energy_j = 0.0
        processing_energy_j = 0.0
        p2p_energy_j = 0.0
        packet_error_probability = 1.0
        packet_success_probability = 0.0
        target_reliability = config.target_reliability
        required_transmission_time_s = 0.0
        required_time_result = RequiredTransmissionTimeResult(
            required_transmission_time_s=0.0,
            required_reliability_met=False,
            required_transmission_time_capped=False,
            success_probability_at_required_time=0.0,
            required_time_search_max_s=config.max_required_transmission_time_s,
        )
        max_attempts = 0
        deadline_delivery_probability = 0.0
        expected_attempts = 0.0
        expected_latency_s = 0.0
        expected_energy_j = 0.0

    return LinkTransmissionRecord(
        scenario_id=channel_record.scenario_id,
        tx_id=channel_record.tx_id,
        rx_id=channel_record.rx_id,
        resource_id=channel_record.resource_id,
        channel_model_id=channel_record.channel_model_id,
        link_transmission_model_id=config.link_transmission_model_id,
        distance_3d_m=channel_record.distance_3d_m,
        payload_bits=config.payload_bits,
        bandwidth_hz=bandwidth_hz,
        spectral_efficiency_bps_per_hz=config.spectral_efficiency_bps_per_hz,
        effective_rate_bps=effective_rate_bps,
        selected=selected,
        active=active,
        transmission_attempted=transmission_attempted,
        propagation_delay_s=propagation_delay_s,
        transmission_delay_s=transmission_delay_s,
        processing_delay_s=processing_delay_s,
        queueing_delay_s=queueing_delay_s,
        p2p_latency_s=p2p_latency_s,
        tx_power_w=tx_power_w,
        rx_circuit_power_w=config.rx_circuit_power_w,
        processing_power_w=config.processing_power_w,
        tx_energy_j=tx_energy_j,
        rx_energy_j=rx_energy_j,
        processing_energy_j=processing_energy_j,
        p2p_energy_j=p2p_energy_j,
        packet_error_probability=packet_error_probability,
        packet_success_probability=packet_success_probability,
        target_reliability=target_reliability,
        required_transmission_time_s=required_transmission_time_s,
        required_reliability_met=required_time_result.required_reliability_met,
        required_transmission_time_capped=required_time_result.required_transmission_time_capped,
        success_probability_at_required_time=(
            required_time_result.success_probability_at_required_time
        ),
        required_time_search_max_s=required_time_result.required_time_search_max_s,
        attempt_duration_s=attempt_duration_s,
        max_attempts_within_deadline=max_attempts,
        deadline_delivery_probability=deadline_delivery_probability,
        expected_attempts=expected_attempts,
        expected_latency_s=expected_latency_s,
        expected_energy_j=expected_energy_j,
        finite_blocklength_regime_id=URLLC_FINITE_BLOCKLENGTH_REGIME_ID,
        link_transmission_regime=URLLC_FINITE_BLOCKLENGTH_REGIME_ID,
    )


def _clip_probability(value: float) -> float:
    if not isfinite(value):
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
