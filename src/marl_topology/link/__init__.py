"""Link models for the clean skeleton."""

from .regime import (
    STAGE2_DETERMINISTIC_DISTANCE_REGIME,
    STAGE2_DETERMINISTIC_DISTANCE_REGIME_ID,
    LinkModelRegime,
    get_stage2_link_regime,
    validate_link_record_against_regime,
)
from .simple_link_model import LinkRecord, SimpleLinkModel
from .transmission import (
    LinkTransmissionConfig,
    LinkTransmissionRecord,
    RequiredTransmissionTimeResult,
    URLLC_FINITE_BLOCKLENGTH_REGIME_ID,
    dbm_to_watt,
    deadline_retransmission_probability,
    evaluate_link_transmission,
    finite_blocklength_packet_error_probability,
    finite_blocklength_packet_success_probability,
    required_transmission_time_for_reliability,
    sinr_db_to_linear,
    solve_required_transmission_time_for_reliability,
)
from .transmission_fixtures import (
    LinkTransmissionFixture,
    get_link_transmission_fixture,
    iter_link_transmission_fixtures,
)

__all__ = [
    "LinkTransmissionConfig",
    "LinkTransmissionFixture",
    "LinkTransmissionRecord",
    "RequiredTransmissionTimeResult",
    "LinkModelRegime",
    "LinkRecord",
    "STAGE2_DETERMINISTIC_DISTANCE_REGIME",
    "STAGE2_DETERMINISTIC_DISTANCE_REGIME_ID",
    "SimpleLinkModel",
    "URLLC_FINITE_BLOCKLENGTH_REGIME_ID",
    "dbm_to_watt",
    "deadline_retransmission_probability",
    "evaluate_link_transmission",
    "finite_blocklength_packet_error_probability",
    "finite_blocklength_packet_success_probability",
    "get_link_transmission_fixture",
    "get_stage2_link_regime",
    "iter_link_transmission_fixtures",
    "required_transmission_time_for_reliability",
    "sinr_db_to_linear",
    "solve_required_transmission_time_for_reliability",
    "validate_link_record_against_regime",
]
