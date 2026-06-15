"""Deterministic Stage 3.3 link transmission fixtures."""

from __future__ import annotations

from dataclasses import dataclass

from marl_topology.channel import evaluate_channel, get_channel_model_fixture

from .transmission import (
    LinkTransmissionConfig,
    LinkTransmissionRecord,
    evaluate_link_transmission,
)


@dataclass(frozen=True, slots=True)
class LinkTransmissionFixture:
    fixture_id: str
    description: str
    config: LinkTransmissionConfig
    selected: bool = True
    active: bool = True

    def __post_init__(self) -> None:
        if not self.fixture_id:
            raise ValueError("fixture_id must be non-empty")

    def evaluate(self) -> LinkTransmissionRecord:
        channel_fixture = get_channel_model_fixture("free_space_close")
        channel_record = evaluate_channel(
            channel_fixture.scene,
            channel_fixture.tx_id,
            channel_fixture.rx_id,
        )
        return evaluate_link_transmission(
            channel_record,
            self.config,
            selected=self.selected,
            active=self.active,
        )


def iter_link_transmission_fixtures() -> tuple[LinkTransmissionFixture, ...]:
    return (
        LinkTransmissionFixture(
            fixture_id="payload_small",
            description="Small payload point-to-point transmission.",
            config=LinkTransmissionConfig(payload_bits=4_000),
        ),
        LinkTransmissionFixture(
            fixture_id="payload_large",
            description="Large payload point-to-point transmission.",
            config=LinkTransmissionConfig(payload_bits=40_000),
        ),
        LinkTransmissionFixture(
            fixture_id="bandwidth_narrow",
            description="Narrow bandwidth point-to-point transmission.",
            config=LinkTransmissionConfig(bandwidth_hz=5e6),
        ),
        LinkTransmissionFixture(
            fixture_id="bandwidth_wide",
            description="Wide bandwidth point-to-point transmission.",
            config=LinkTransmissionConfig(bandwidth_hz=20e6),
        ),
        LinkTransmissionFixture(
            fixture_id="unselected_edge",
            description="Unselected edge fixture must consume no transmission energy.",
            config=LinkTransmissionConfig(),
            selected=False,
            active=False,
        ),
    )


def get_link_transmission_fixture(fixture_id: str) -> LinkTransmissionFixture:
    for fixture in iter_link_transmission_fixtures():
        if fixture.fixture_id == fixture_id:
            return fixture
    raise KeyError(f"unknown link transmission fixture: {fixture_id}")
