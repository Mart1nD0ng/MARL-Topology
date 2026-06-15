"""Deterministic Stage 3.2 channel model.

The model produces communication-channel records only. It does not define
consensus reliability, reward, topology optimality, or training behavior.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from functools import lru_cache
from math import exp, floor, log10, sqrt
import random

from marl_topology.geometry3d import LoSState, VisibilityRecord, evaluate_visibility


THERMAL_NOISE_DENSITY_DBM_PER_HZ = -174.0
STAGE3_CHANNEL_REGIME_ID = "stage3_channel_v1_fspl_sinr"


PATH_LOSS_MODEL_FSPL = "fspl_nlos_penalty"
PATH_LOSS_MODEL_UMI = "umi_street_canyon_38901"
PATH_LOSS_MODEL_V2X = "v2x_37885"
# nodes at or below this height are treated as vehicles for the 37.885 V2V/V2I split
V2X_VEHICLE_HEIGHT_THRESHOLD_M = 3.0

# --- TR 37.885 / 36.885 / 38.901 standardized large-scale fading constants ------------
# Shadow fading sigma (dB), TR 37.885 Table 6.2.1-1; the LOS value applies to NLOSv too
# (sec. 6.2.2). NLOS is the urban different-street (building-blocked) state.
V2V_SHADOW_STD_LOS_DB = 3.0
V2V_SHADOW_STD_NLOS_DB = 4.0
# UMi-Street-Canyon shadow fading, TR 38.901 Table 7.4.1-1 (used for non-UE-type links).
UMI_SHADOW_STD_LOS_DB = 4.0
UMI_SHADOW_STD_NLOS_DB = 7.82
# Shadowing decorrelation distance, TR 36.885 Table A.1.4-1 (urban). Exponential
# (Gudmundson) autocorrelation per TR 38.901 sec. 7.4.4 / ITU-R P.1816.
SHADOW_DECORRELATION_DISTANCE_M = 10.0
# NLOSv blocker height: urban antenna Option A = 100% Type-2 passenger cars (vehicle
# height 1.6 m, antenna at roof), TR 37.885 sec. 6.1.2 -- deterministic under Option A.
NLOSV_BLOCKER_HEIGHT_M = 1.6
# Nodes up to this height are "UE-type" for the 37.885 link-type mapping: V2V and
# UE-type-RSU (5 m) links share the V2V channel model (Table 6.2.1-2); only BS-type
# infrastructure (25 m) would use the macro models.
V2X_UE_TYPE_HEIGHT_MAX_M = 10.0


@dataclass(frozen=True, slots=True)
class ChannelModelConfig:
    channel_model_id: str = STAGE3_CHANNEL_REGIME_ID
    carrier_frequency_hz: float = 5.9e9
    bandwidth_hz: float = 10e6
    default_tx_power_dbm: float = 20.0
    tx_antenna_gain_db: float = 0.0
    rx_antenna_gain_db: float = 0.0
    noise_figure_db: float = 7.0
    nlos_penalty_db: float = 20.0
    shadowing_std_db: float = 0.0
    min_distance_m: float = 1.0
    no_interference_power_dbm: float = -300.0
    # path_loss_model selects the propagation formula (a calibration knob):
    # - "fspl_nlos_penalty" (default, byte-identical): free-space path loss + a flat
    #   nlos_penalty_db when ray-box visibility reports NLOS.
    # - "umi_street_canyon_38901": 3GPP TR 38.901 UMi-Street-Canyon LOS/NLOS path loss
    #   (distance-dependent NLOS exponent 35.3*log10(d), embodied NLOS penalty -- the
    #   standardized urban-canyon model; applied to all links with the actual node heights
    #   as an approximation, documented for the calibration study).
    # - "v2x_37885" (the calibrated V2X choice): TR 37.885 urban V2V LOS/NLOS for
    #   vehicle-to-vehicle links (both endpoints at vehicle height), TR 38.901 UMi for
    #   V2I links -- the standards-based model pair for urban V2X sidelink.
    path_loss_model: str = PATH_LOSS_MODEL_FSPL
    # shadowing_37885 (opt-in, requires v2x_37885): standardized log-normal shadow fading.
    # Per-state sigma (V2V/UE-type: LOS & NLOSv 3 dB, NLOS 4 dB per TR 37.885; UMi links:
    # 4 / 7.82 dB per TR 38.901), RECIPROCAL, and spatially correlated: each endpoint reads
    # a seeded unit-variance lattice Gaussian field (cell = the 10 m decorrelation distance,
    # the position-form equivalent of the TR 36.885 exponential update), and the link value
    # is sigma * (g_tx + g_rx) / sqrt(2). The field is keyed by the scene (crc32 of
    # scenario_id unless an explicit shadowing_seed is passed) and by POSITION only, so a
    # moving vehicle sees temporally correlated shadowing across trajectory frames.
    shadowing_37885: bool = False
    # nlosv_37885 (opt-in, requires v2x_37885): the TR 37.885 sec. 6.2 channel-state model.
    # Building-blocked links stay NLOS. Building-clear UE-type links draw the LOS/NLOSv
    # state stochastically -- P(LOS) = min(1, 1.05*exp(-0.0114*d)) (urban, Table 6.2-1) --
    # with a PERSISTENT per-pair uniform (the baseline does not re-draw the state), and an
    # NLOSv link adds max(0 dB, Normal(mu_a, sigma_a)) vehicle-blockage loss per sec. 6.2.1
    # (Option A blocker height 1.6 m; antennas below the blocker -> mu 9 dB, straddling ->
    # mu 5 dB, both above -> 0; + max(0, 15*log10(d)-41) distance term; persistent per-pair
    # normal draw). Also applies the Table 6.2.1-2 link-type mapping: UE-type RSU (5 m)
    # links use the V2V path-loss family instead of UMi. NLOSv loss is reported in
    # los_penalty_db (the los_state field keeps the building-visibility LOS/NLOS truth).
    nlosv_37885: bool = False
    # Robustness draws: different realizations re-sample every stochastic large-scale term
    # (shadow field, NLOSv states and losses) for the same scene -- M-draw distributional
    # feasibility evaluates the same topology under realizations 0..M-1.
    shadowing_realization: int = 0

    def __post_init__(self) -> None:
        if not self.channel_model_id:
            raise ValueError("channel_model_id must be non-empty")
        if self.carrier_frequency_hz <= 0:
            raise ValueError("carrier_frequency_hz must be positive")
        if self.bandwidth_hz <= 0:
            raise ValueError("bandwidth_hz must be positive")
        if self.nlos_penalty_db < 0:
            raise ValueError("nlos_penalty_db must be nonnegative")
        if self.shadowing_std_db < 0:
            raise ValueError("shadowing_std_db must be nonnegative")
        if self.min_distance_m <= 0:
            raise ValueError("min_distance_m must be positive")
        if self.path_loss_model not in (
            PATH_LOSS_MODEL_FSPL,
            PATH_LOSS_MODEL_UMI,
            PATH_LOSS_MODEL_V2X,
        ):
            raise ValueError(f"unknown path_loss_model: {self.path_loss_model}")
        if (self.shadowing_37885 or self.nlosv_37885) and self.path_loss_model != PATH_LOSS_MODEL_V2X:
            raise ValueError("shadowing_37885 / nlosv_37885 require path_loss_model v2x_37885")
        if self.shadowing_37885 and self.shadowing_std_db > 0.0:
            raise ValueError("shadowing_37885 replaces the legacy shadowing_std_db term")


@dataclass(frozen=True, slots=True)
class ActiveTransmission:
    tx_id: str
    rx_id: str
    resource_id: str = "resource_0"
    tx_power_dbm: float | None = None
    active: bool = True

    def __post_init__(self) -> None:
        if not self.tx_id or not self.rx_id:
            raise ValueError("tx_id and rx_id must be non-empty")
        if self.tx_id == self.rx_id:
            raise ValueError("tx_id and rx_id must differ")
        if not self.resource_id:
            raise ValueError("resource_id must be non-empty")


@dataclass(frozen=True, slots=True)
class ChannelRecord:
    scenario_id: str
    tx_id: str
    rx_id: str
    resource_id: str
    channel_model_id: str
    distance_3d_m: float
    los_state: LoSState
    blocker_ids: tuple[str, ...]
    carrier_frequency_hz: float
    bandwidth_hz: float
    base_path_loss_db: float
    los_penalty_db: float
    shadowing_db: float
    path_loss_db: float
    tx_power_dbm: float
    tx_antenna_gain_db: float
    rx_antenna_gain_db: float
    rx_power_dbm: float
    noise_power_dbm: float
    interference_power_dbm: float
    interference_tx_ids: tuple[str, ...]
    sinr_db: float
    shadowing_seed: int | None
    visibility_regime: str

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must be non-empty")
        if not self.tx_id or not self.rx_id:
            raise ValueError("tx_id and rx_id must be non-empty")
        if self.tx_id == self.rx_id:
            raise ValueError("tx_id and rx_id must differ")
        if not self.resource_id:
            raise ValueError("resource_id must be non-empty")
        if not self.channel_model_id:
            raise ValueError("channel_model_id must be non-empty")
        if self.distance_3d_m < 0:
            raise ValueError("distance_3d_m must be nonnegative")
        if self.carrier_frequency_hz <= 0 or self.bandwidth_hz <= 0:
            raise ValueError("frequency and bandwidth must be positive")
        if self.los_state == LoSState.LOS and self.blocker_ids:
            raise ValueError("LOS channel record cannot include blockers")
        if self.los_state == LoSState.NLOS and not self.blocker_ids:
            raise ValueError("NLOS channel record must include blockers")


def dbm_to_mw(power_dbm: float) -> float:
    return 10.0 ** (power_dbm / 10.0)


def mw_to_dbm(power_mw: float, floor_dbm: float = -300.0) -> float:
    if power_mw <= 0.0:
        return floor_dbm
    return 10.0 * log10(power_mw)


def free_space_path_loss_db(
    distance_3d_m: float,
    carrier_frequency_hz: float,
    min_distance_m: float = 1.0,
) -> float:
    if distance_3d_m < 0:
        raise ValueError("distance_3d_m must be nonnegative")
    if carrier_frequency_hz <= 0:
        raise ValueError("carrier_frequency_hz must be positive")
    if min_distance_m <= 0:
        raise ValueError("min_distance_m must be positive")
    effective_distance_m = max(distance_3d_m, min_distance_m)
    return 20.0 * log10(effective_distance_m) + 20.0 * log10(carrier_frequency_hz) - 147.55


def noise_power_dbm(bandwidth_hz: float, noise_figure_db: float) -> float:
    if bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be positive")
    return THERMAL_NOISE_DENSITY_DBM_PER_HZ + 10.0 * log10(bandwidth_hz) + noise_figure_db


def evaluate_channel(
    scene,
    tx_id: str,
    rx_id: str,
    config: ChannelModelConfig | None = None,
    active_transmissions: tuple[ActiveTransmission, ...] = (),
    resource_id: str = "resource_0",
    tx_power_dbm: float | None = None,
    shadowing_seed: int | None = None,
) -> ChannelRecord:
    if config is None:
        config = ChannelModelConfig()
    if not resource_id:
        raise ValueError("resource_id must be non-empty")

    visibility = evaluate_visibility(scene, tx_id, rx_id)
    desired_tx_power_dbm = (
        config.default_tx_power_dbm if tx_power_dbm is None else tx_power_dbm
    )
    desired_terms = _channel_terms(
        visibility=visibility,
        config=config,
        tx_power_dbm=desired_tx_power_dbm,
        shadowing_seed=shadowing_seed,
    )

    interference_power_mw = 0.0
    interference_tx_ids: set[str] = set()
    for transmission in active_transmissions:
        if not transmission.active:
            continue
        if transmission.resource_id != resource_id:
            continue
        if transmission.tx_id == tx_id:
            continue
        interference_visibility = evaluate_visibility(scene, transmission.tx_id, rx_id)
        interference_tx_power_dbm = (
            config.default_tx_power_dbm
            if transmission.tx_power_dbm is None
            else transmission.tx_power_dbm
        )
        interference_terms = _channel_terms(
            visibility=interference_visibility,
            config=config,
            tx_power_dbm=interference_tx_power_dbm,
            shadowing_seed=shadowing_seed,
        )
        interference_power_mw += dbm_to_mw(interference_terms["rx_power_dbm"])
        interference_tx_ids.add(transmission.tx_id)

    interference_power_dbm = mw_to_dbm(
        interference_power_mw,
        floor_dbm=config.no_interference_power_dbm,
    )
    noise_power = noise_power_dbm(config.bandwidth_hz, config.noise_figure_db)
    signal_power_mw = dbm_to_mw(desired_terms["rx_power_dbm"])
    denominator_mw = dbm_to_mw(noise_power) + interference_power_mw
    sinr_db = mw_to_dbm(signal_power_mw / denominator_mw)

    return ChannelRecord(
        scenario_id=visibility.scenario_id,
        tx_id=tx_id,
        rx_id=rx_id,
        resource_id=resource_id,
        channel_model_id=config.channel_model_id,
        distance_3d_m=visibility.distance_3d_m,
        los_state=visibility.los_state,
        blocker_ids=tuple(blocker.building_id for blocker in visibility.blocker_records),
        carrier_frequency_hz=config.carrier_frequency_hz,
        bandwidth_hz=config.bandwidth_hz,
        base_path_loss_db=desired_terms["base_path_loss_db"],
        los_penalty_db=desired_terms["los_penalty_db"],
        shadowing_db=desired_terms["shadowing_db"],
        path_loss_db=desired_terms["path_loss_db"],
        tx_power_dbm=desired_tx_power_dbm,
        tx_antenna_gain_db=config.tx_antenna_gain_db,
        rx_antenna_gain_db=config.rx_antenna_gain_db,
        rx_power_dbm=desired_terms["rx_power_dbm"],
        noise_power_dbm=noise_power,
        interference_power_dbm=interference_power_dbm,
        interference_tx_ids=tuple(sorted(interference_tx_ids)),
        sinr_db=sinr_db,
        shadowing_seed=shadowing_seed,
        visibility_regime=visibility.visibility_regime,
    )


def umi_street_canyon_path_loss_db(
    distance_3d_m: float,
    carrier_frequency_hz: float,
    *,
    is_nlos: bool,
    h_tx_m: float,
    h_rx_m: float,
    min_distance_m: float = 1.0,
) -> float:
    """3GPP TR 38.901 UMi-Street-Canyon path loss (Table 7.4.1-1).

    LOS: 32.4 + 21*log10(d3D) + 20*log10(fc_GHz) below the breakpoint, with the 40*log10
    slope and breakpoint correction beyond it. NLOS: max(PL_LOS, 35.3*log10(d3D) + 22.4 +
    21.3*log10(fc_GHz) - 0.3*(hUT - 1.5)) -- a distance-dependent NLOS exponent instead of
    the flat 20 dB penalty. Applied with the actual node heights (RSU 5 m / vehicle 1.5 m)
    as a documented approximation of the BS/UT roles for the calibration study."""
    distance = max(distance_3d_m, min_distance_m, 1.0)
    fc_ghz = carrier_frequency_hz / 1e9
    h_bs = max(h_tx_m, h_rx_m)
    h_ut = min(h_tx_m, h_rx_m)
    # breakpoint distance with effective antenna heights (h - 1.0 m environment height)
    h_bs_eff = max(h_bs - 1.0, 0.1)
    h_ut_eff = max(h_ut - 1.0, 0.1)
    d_bp = 4.0 * h_bs_eff * h_ut_eff * carrier_frequency_hz / 299_792_458.0
    if distance <= d_bp or d_bp <= 0.0:
        pl_los = 32.4 + 21.0 * log10(distance) + 20.0 * log10(fc_ghz)
    else:
        pl_los = (
            32.4
            + 40.0 * log10(distance)
            + 20.0 * log10(fc_ghz)
            - 9.5 * log10(d_bp ** 2 + (h_bs - h_ut) ** 2)
        )
    if not is_nlos:
        return pl_los
    pl_nlos = 35.3 * log10(distance) + 22.4 + 21.3 * log10(fc_ghz) - 0.3 * (h_ut - 1.5)
    return max(pl_los, pl_nlos)


def v2v_37885_path_loss_db(
    distance_3d_m: float,
    carrier_frequency_hz: float,
    *,
    is_nlos: bool,
    min_distance_m: float = 1.0,
) -> float:
    """3GPP TR 37.885 urban V2V path loss (Table 6.2.1-1).

    LOS: 38.77 + 16.7*log10(d3D) + 18.2*log10(fc_GHz). NLOS (urban crossing):
    36.85 + 30*log10(d3D) + 18.9*log10(fc_GHz). The NLOSv state uses the LOS equation;
    its stochastic vehicle-blockage extra loss is the opt-in nlosv_37885 term added in
    _channel_terms (off by default)."""
    distance = max(distance_3d_m, min_distance_m, 1.0)
    fc_ghz = carrier_frequency_hz / 1e9
    if is_nlos:
        return 36.85 + 30.0 * log10(distance) + 18.9 * log10(fc_ghz)
    return 38.77 + 16.7 * log10(distance) + 18.2 * log10(fc_ghz)


def _scene_shadowing_seed(
    visibility: VisibilityRecord,
    config: ChannelModelConfig,
    shadowing_seed: int | None,
) -> str:
    """Stable per-scene stochastic key for the 37.885 large-scale terms.

    Defaults to crc32(scenario_id) so every evaluation path (link records, network
    records, MAC power tables) sees the SAME realization without seed threading, and
    trajectory frames (advance_scene keeps scenario_id) stay on the same shadow field.
    shadowing_realization indexes independent robustness draws."""
    base = (
        zlib.crc32(visibility.scenario_id.encode("utf-8"))
        if shadowing_seed is None
        else shadowing_seed
    )
    return f"{base}r{config.shadowing_realization}"


def _pair_uniform(seed: str, visibility: VisibilityRecord, tag: str) -> float:
    a, b = sorted((visibility.tx_id, visibility.rx_id))
    return random.Random(f"{seed}:{tag}:{a}:{b}").random()


def _pair_gauss(seed: str, visibility: VisibilityRecord, tag: str) -> float:
    a, b = sorted((visibility.tx_id, visibility.rx_id))
    return random.Random(f"{seed}:{tag}:{a}:{b}").gauss(0.0, 1.0)


def _nlosv_state(visibility: VisibilityRecord, seed: str) -> bool:
    """TR 37.885 Table 6.2-1 urban: P(LOS) = min(1, 1.05*exp(-0.0114*d)), persistent
    per-pair uniform (the baseline does not re-draw the LOS/NLOSv state)."""
    distance = max(visibility.distance_3d_m, 1.0)
    p_los = min(1.0, 1.05 * exp(-0.0114 * distance))
    return _pair_uniform(seed, visibility, "nlosv_state") > p_los


def _nlosv_blockage_loss_db(visibility: VisibilityRecord, seed: str) -> float:
    """TR 37.885 sec. 6.2.1 NLOSv additional vehicle-blockage loss.

    max(0 dB, Normal(mu_a, sigma_a)) with the antenna-vs-blocker height cases; Option A
    blocker height (1.6 m). The standard-normal draw is persistent per pair, so the loss
    evolves smoothly with distance along a trajectory instead of resampling each frame."""
    h_min = min(visibility.ray_start_m.z_m, visibility.ray_end_m.z_m)
    h_max = max(visibility.ray_start_m.z_m, visibility.ray_end_m.z_m)
    if h_min > NLOSV_BLOCKER_HEIGHT_M:
        return 0.0
    distance = max(visibility.distance_3d_m, 1.0)
    distance_term = max(0.0, 15.0 * log10(distance) - 41.0)
    if h_max < NLOSV_BLOCKER_HEIGHT_M:
        mu_a, sigma_a = 9.0 + distance_term, 4.5
    else:
        mu_a, sigma_a = 5.0 + distance_term, 4.0
    return max(0.0, mu_a + sigma_a * _pair_gauss(seed, visibility, "nlosv_loss"))


@lru_cache(maxsize=262144)
def _shadow_field_corner(seed: str, ix: int, iy: int) -> float:
    return random.Random(f"{seed}:shadowfield:{ix}:{iy}").gauss(0.0, 1.0)


def _shadow_field_value(seed: str, x_m: float, y_m: float) -> float:
    """Unit-variance lattice Gaussian field with cell size = the 10 m decorrelation
    distance: normalized bilinear interpolation of seeded corner normals. Correlation
    decays to zero beyond one cell -- the position-form (Gudmundson-type) counterpart of
    the TR 36.885 exponential shadowing update, and a pure function of position, so
    vehicle motion yields temporally correlated shadowing."""
    gx = x_m / SHADOW_DECORRELATION_DISTANCE_M
    gy = y_m / SHADOW_DECORRELATION_DISTANCE_M
    ix, iy = floor(gx), floor(gy)
    fx, fy = gx - ix, gy - iy
    weights = (
        ((1.0 - fx) * (1.0 - fy), ix, iy),
        (fx * (1.0 - fy), ix + 1, iy),
        ((1.0 - fx) * fy, ix, iy + 1),
        (fx * fy, ix + 1, iy + 1),
    )
    value = sum(w * _shadow_field_corner(seed, cx, cy) for w, cx, cy in weights)
    norm = sqrt(sum(w * w for w, _cx, _cy in weights))
    return value / norm if norm > 0.0 else 0.0


def _shadowing_37885_db(
    visibility: VisibilityRecord,
    seed: str,
    is_v2v: bool,
    link_state: str,
) -> float:
    """Reciprocal, spatially correlated log-normal shadowing with per-state sigma."""
    if is_v2v:
        sigma = V2V_SHADOW_STD_NLOS_DB if link_state == "nlos" else V2V_SHADOW_STD_LOS_DB
    else:
        sigma = UMI_SHADOW_STD_NLOS_DB if link_state == "nlos" else UMI_SHADOW_STD_LOS_DB
    g_tx = _shadow_field_value(seed, visibility.ray_start_m.x_m, visibility.ray_start_m.y_m)
    g_rx = _shadow_field_value(seed, visibility.ray_end_m.x_m, visibility.ray_end_m.y_m)
    return sigma * (g_tx + g_rx) / sqrt(2.0)


def _channel_terms(
    visibility: VisibilityRecord,
    config: ChannelModelConfig,
    tx_power_dbm: float,
    shadowing_seed: int | None,
) -> dict[str, float]:
    if config.path_loss_model == PATH_LOSS_MODEL_V2X:
        h_tx = visibility.ray_start_m.z_m
        h_rx = visibility.ray_end_m.z_m
        if config.nlosv_37885:
            # Table 6.2.1-2 link-type mapping: all UE-type links (vehicles and 5 m
            # roadside RSUs) share the V2V channel family.
            is_v2v = (
                h_tx <= V2X_UE_TYPE_HEIGHT_MAX_M and h_rx <= V2X_UE_TYPE_HEIGHT_MAX_M
            )
        else:
            is_v2v = (
                h_tx <= V2X_VEHICLE_HEIGHT_THRESHOLD_M and h_rx <= V2X_VEHICLE_HEIGHT_THRESHOLD_M
            )
        los_penalty = 0.0  # the standards formulas embody the NLOS penalty
        building_nlos = visibility.los_state == LoSState.NLOS
        link_state = "nlos" if building_nlos else "los"
        if is_v2v:
            base_path_loss = v2v_37885_path_loss_db(
                visibility.distance_3d_m,
                config.carrier_frequency_hz,
                is_nlos=building_nlos,
                min_distance_m=config.min_distance_m,
            )
            if config.nlosv_37885 and not building_nlos:
                scene_seed = _scene_shadowing_seed(visibility, config, shadowing_seed)
                if _nlosv_state(visibility, scene_seed):
                    link_state = "nlosv"
                    los_penalty = _nlosv_blockage_loss_db(visibility, scene_seed)
        else:
            base_path_loss = umi_street_canyon_path_loss_db(
                visibility.distance_3d_m,
                config.carrier_frequency_hz,
                is_nlos=building_nlos,
                h_tx_m=h_tx,
                h_rx_m=h_rx,
                min_distance_m=config.min_distance_m,
            )
        if config.shadowing_37885:
            scene_seed = _scene_shadowing_seed(visibility, config, shadowing_seed)
            shadowing = _shadowing_37885_db(visibility, scene_seed, is_v2v, link_state)
            path_loss = base_path_loss + los_penalty + shadowing
            rx_power_dbm = (
                tx_power_dbm
                + config.tx_antenna_gain_db
                + config.rx_antenna_gain_db
                - path_loss
            )
            return {
                "base_path_loss_db": base_path_loss,
                "los_penalty_db": los_penalty,
                "shadowing_db": shadowing,
                "path_loss_db": path_loss,
                "rx_power_dbm": rx_power_dbm,
            }
    elif config.path_loss_model == PATH_LOSS_MODEL_UMI:
        base_path_loss = umi_street_canyon_path_loss_db(
            visibility.distance_3d_m,
            config.carrier_frequency_hz,
            is_nlos=visibility.los_state == LoSState.NLOS,
            h_tx_m=visibility.ray_start_m.z_m,
            h_rx_m=visibility.ray_end_m.z_m,
            min_distance_m=config.min_distance_m,
        )
        los_penalty = 0.0  # the UMi NLOS formula embodies the penalty
    else:
        base_path_loss = free_space_path_loss_db(
            visibility.distance_3d_m,
            config.carrier_frequency_hz,
            config.min_distance_m,
        )
        los_penalty = config.nlos_penalty_db if visibility.los_state == LoSState.NLOS else 0.0
    shadowing = _shadowing_db(visibility, config, shadowing_seed)
    path_loss = base_path_loss + los_penalty + shadowing
    rx_power_dbm = (
        tx_power_dbm
        + config.tx_antenna_gain_db
        + config.rx_antenna_gain_db
        - path_loss
    )
    return {
        "base_path_loss_db": base_path_loss,
        "los_penalty_db": los_penalty,
        "shadowing_db": shadowing,
        "path_loss_db": path_loss,
        "rx_power_dbm": rx_power_dbm,
    }


def _shadowing_db(
    visibility: VisibilityRecord,
    config: ChannelModelConfig,
    shadowing_seed: int | None,
) -> float:
    if config.shadowing_std_db == 0.0:
        return 0.0
    if shadowing_seed is None:
        raise ValueError("shadowing_seed is required when shadowing_std_db is positive")
    seed_text = (
        f"{shadowing_seed}:{visibility.scenario_id}:"
        f"{visibility.tx_id}:{visibility.rx_id}:{visibility.los_state.value}"
    )
    rng = random.Random(seed_text)
    return rng.gauss(0.0, config.shadowing_std_db)
