"""Stage 3.4 network communication records and fixtures."""

from .communication import (
    NetworkCommunicationConfig,
    NetworkCommunicationRecord,
    NetworkHopRecord,
    NetworkTransmissionSpec,
    evaluate_network_communication,
)
from .fixtures import (
    NetworkCommunicationFixture,
    get_network_communication_fixture,
    iter_network_communication_fixtures,
)

__all__ = [
    "NetworkCommunicationConfig",
    "NetworkCommunicationFixture",
    "NetworkCommunicationRecord",
    "NetworkHopRecord",
    "NetworkTransmissionSpec",
    "evaluate_network_communication",
    "get_network_communication_fixture",
    "iter_network_communication_fixtures",
]
