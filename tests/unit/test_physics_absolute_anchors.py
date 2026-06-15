"""Absolute-value anchor tests for the channel + finite-blocklength physics.

The rest of the physics suite only checks RELATIVE behavior (A > B monotonicity),
so a wrong constant or a dB/linear/sign error could shift the absolute SINR by tens
of dB while every relative test stayed green -- silently relocating the feasibility
boundary the agent trains against. These tests pin a few ABSOLUTE values computed
from first principles (textbook formulas), so the magic constants (-147.55 FSPL,
-174.0 noise floor, the Polyanskiy-Poor-Verdu finite-blocklength approximation)
can never drift unnoticed.

Every expected value below is derived by hand from the standard formula, NOT read
back from the code, so the check is non-circular. (Independently recomputed and
confirmed to full float precision against the live functions.)
"""

import math

from marl_topology.channel.model import free_space_path_loss_db, noise_power_dbm
from marl_topology.link.transmission import (
    finite_blocklength_packet_success_probability,
)


def test_free_space_path_loss_absolute_value_100m_5p9ghz() -> None:
    # Friis FSPL (f in Hz): PL = 20log10(d) + 20log10(f) + 20log10(4*pi/c).
    # 20log10(100)=40.000 ; 20log10(5.9e9)=195.41704 ; 20log10(4*pi/c)=-147.5522.
    # Sum = 87.8648 dB (textbook-exact). The code uses the 2-dp constant -147.55,
    # giving 87.867 dB; the 0.0022 dB gap is the constant rounding, well within 0.01.
    pl = free_space_path_loss_db(distance_3d_m=100.0, carrier_frequency_hz=5.9e9)
    assert math.isclose(pl, 87.867, abs_tol=0.01), pl
    # the lumped Friis constant must stay the standard -147.55(22), not e.g. the
    # +32.45 (MHz/km) form -- a wrong convention would shift this by ~180 dB.
    assert -147.6 < (pl - (20.0 * math.log10(100.0) + 20.0 * math.log10(5.9e9))) < -147.5


def test_thermal_noise_power_absolute_value_10mhz_nf7() -> None:
    # kTB + NF: N = -174 dBm/Hz + 10log10(B) + NF = -174 + 70 + 7 = -97.0 dBm.
    noise = noise_power_dbm(bandwidth_hz=10e6, noise_figure_db=7.0)
    assert math.isclose(noise, -97.0, abs_tol=0.001), noise
    # the -174 dBm/Hz density is the IEEE-rounded kT at 290 K (exact -173.975).
    assert math.isclose(noise_power_dbm(bandwidth_hz=1.0, noise_figure_db=0.0), -174.0, abs_tol=0.001)


def test_link_budget_sinr_absolute_value_los_no_interference() -> None:
    # Link budget with no interference: SINR_dB = Prx_dBm - N_dBm,
    # Prx_dBm = Ptx + Gtx + Grx - PL. With Ptx=20 dBm, gains 0, d=100 m, f=5.9 GHz,
    # B=10 MHz, NF=7: Prx = 20 - 87.867 = -67.867 dBm ; SINR = -67.867 - (-97.0) = 29.133 dB.
    tx_power_dbm = 20.0
    rx_power_dbm = tx_power_dbm - free_space_path_loss_db(
        distance_3d_m=100.0, carrier_frequency_hz=5.9e9
    )
    assert math.isclose(rx_power_dbm, -67.867, abs_tol=0.02), rx_power_dbm
    sinr_db = rx_power_dbm - noise_power_dbm(bandwidth_hz=10e6, noise_figure_db=7.0)
    assert math.isclose(sinr_db, 29.133, abs_tol=0.02), sinr_db


def test_finite_blocklength_psucc_saturates_to_one_at_high_sinr() -> None:
    # n=B*t=12000 symbols, C=log2(1+1000)=9.967 bits/sym, n*C=119607 >> L=12000,
    # so the Q-argument -> +inf and PER -> 0, psucc -> 1.
    psucc = finite_blocklength_packet_success_probability(
        sinr_db=30.0, bandwidth_hz=10e6, transmission_time_s=0.0012, payload_bits=12_000
    )
    assert psucc >= 1.0 - 1e-9, psucc


def test_finite_blocklength_psucc_collapses_to_zero_at_low_sinr() -> None:
    # gamma=0.1, C=0.1375, n*C=1650 << L=12000, so the Q-argument -> -inf,
    # PER -> 1, psucc -> 0.
    psucc = finite_blocklength_packet_success_probability(
        sinr_db=-10.0, bandwidth_hz=10e6, transmission_time_s=0.0012, payload_bits=12_000
    )
    assert psucc <= 1e-9, psucc


def test_finite_blocklength_psucc_borderline_at_zero_db() -> None:
    # The single most diagnostic anchor: at SINR=0 dB, C=1 so n*C=12000=L exactly.
    # The only thing keeping psucc off 0.5 is the +0.5*log2(n) correction and the
    # channel dispersion V -- any error in either shifts this visibly. Expected 0.520.
    psucc = finite_blocklength_packet_success_probability(
        sinr_db=0.0, bandwidth_hz=10e6, transmission_time_s=0.0012, payload_bits=12_000
    )
    assert math.isclose(psucc, 0.520, abs_tol=0.005), psucc
