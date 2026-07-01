"""DF1 (Decision-Focused, goal 2): the shadow-fading decorrelation-distance knob.

Goal 2 exposes ``SHADOW_DECORRELATION_DISTANCE_M`` (a hardcoded 10 m constant, TR 36.885 urban) as a
config-threaded, swept knob so the decision-critical psucc gets tunable TEMPORAL autocorrelation (governing
ratio v*dt/d_corr). These are the FAILING-FIRST load-bearing tests for DF1: on HEAD the knob does not exist
(``ChannelModelConfig`` has no ``shadow_decorrelation_distance_m`` field and ``_shadow_field_value`` takes no
``decorr_m``), so every test here fails until the knob is threaded.

Priority order (validation_design v2 A.2): byte-identical-off, validation, field-changes-with-decorr, and the
CORE claim -- larger d_corr RAISES spatial (hence temporal) correlation. The statistical env-property gates
(psucc-marginal-invariance across d_corr, and rho(psucc_t, psucc_{t-1}) monotonicity over trajectory frames,
field-seed averaged) live in the DF1 diagnostic probe (scripts/diagnostics/df1_decorr_env_probe.py), not here.
"""

from statistics import mean

from marl_topology.channel import ChannelModelConfig, evaluate_channel
from marl_topology.channel.model import (
    PATH_LOSS_MODEL_V2X,
    SHADOW_DECORRELATION_DISTANCE_M,
    _shadow_field_value,
)
from marl_topology.data.stage31_scenario_generator import PhysicsRegime
from marl_topology.geometry3d import Point3D
from marl_topology.scenario.scene import Node3D, NodeKind, Scene3D

import pytest


def _scene(*nodes: Node3D, scenario_id: str = "df1_decorr") -> Scene3D:
    return Scene3D(scenario_id=scenario_id, nodes=tuple(nodes))


def _veh(node_id: str, x: float, y: float = 0.0) -> Node3D:
    return Node3D(node_id, NodeKind.VEHICLE, Point3D(x, y, 1.5))


def test_default_decorr_is_the_standards_constant_and_byte_identical() -> None:
    """The knob DEFAULTS to the 10 m TR 36.885 urban constant, so enabling the field with the default is
    byte-identical to HEAD: the config with no decorr arg and the config with decorr=10.0 produce the same
    shadowing_db (and it is active, not 0)."""
    scene = _scene(_veh("veh_0", 0.0), _veh("veh_1", 80.0))
    default = ChannelModelConfig(path_loss_model=PATH_LOSS_MODEL_V2X, shadowing_37885=True)
    explicit = ChannelModelConfig(
        path_loss_model=PATH_LOSS_MODEL_V2X, shadowing_37885=True,
        shadow_decorrelation_distance_m=SHADOW_DECORRELATION_DISTANCE_M,
    )
    assert default.shadow_decorrelation_distance_m == SHADOW_DECORRELATION_DISTANCE_M
    d = evaluate_channel(scene, "veh_0", "veh_1", config=default).shadowing_db
    e = evaluate_channel(scene, "veh_0", "veh_1", config=explicit).shadowing_db
    assert d == e                       # byte-identical at the default
    assert d != 0.0                     # shadowing is genuinely active


def test_decorr_validates_positive() -> None:
    with pytest.raises(ValueError):
        ChannelModelConfig(path_loss_model=PATH_LOSS_MODEL_V2X, shadow_decorrelation_distance_m=0.0)
    with pytest.raises(ValueError):
        ChannelModelConfig(path_loss_model=PATH_LOSS_MODEL_V2X, shadow_decorrelation_distance_m=-5.0)


def test_field_value_changes_with_decorr() -> None:
    """The raw field is re-scaled by the knob: at a position that is a non-integer number of cells from the
    origin, a different cell size yields a different interpolated value (no lru_cache collision: the corner
    normals are cell-size-independent, only the position->lattice mapping changes)."""
    a = _shadow_field_value("seed", 15.0, 7.0, decorr_m=10.0)
    b = _shadow_field_value("seed", 15.0, 7.0, decorr_m=50.0)
    assert a != b


def test_default_decorr_arg_matches_module_constant() -> None:
    """Omitting decorr_m reproduces the module-constant path exactly (byte-identity for the existing callers)."""
    assert _shadow_field_value("seed", 23.0, 4.0) == _shadow_field_value(
        "seed", 23.0, 4.0, decorr_m=SHADOW_DECORRELATION_DISTANCE_M
    )


def test_larger_decorr_raises_spatial_correlation() -> None:
    """THE CORE CLAIM (goal 2): larger decorrelation distance => slower spatial decorrelation => a pair of
    points a fixed 20 m apart is MORE correlated (smaller absolute field difference) at d_corr=100 than at
    d_corr=10. Averaged over positions and field seeds to be non-flaky. This is the spatial proxy for the
    temporal autocorrelation a moving vehicle sees; the trajectory-based rho monotonicity is in the probe."""
    sep = 20.0
    seeds = [f"s{k}" for k in range(6)]
    bases = [0.0, 37.0, 111.0, 208.0, 333.0, 512.0]

    def mean_abs_diff(decorr: float) -> float:
        diffs = [
            abs(_shadow_field_value(s, b, 0.0, decorr_m=decorr)
                - _shadow_field_value(s, b + sep, 0.0, decorr_m=decorr))
            for s in seeds for b in bases
        ]
        return mean(diffs)

    close = mean_abs_diff(100.0)
    far = mean_abs_diff(10.0)
    assert close < far                  # larger d_corr -> more spatially correlated -> smaller step


def test_channel_record_shadowing_changes_with_decorr() -> None:
    """The knob threads all the way to the ChannelRecord: two decorr values give different shadowing_db on the
    same link (proves it is wired through _channel_terms, not just the field helper)."""
    scene = _scene(_veh("veh_0", 0.0), _veh("veh_1", 63.0))
    c10 = ChannelModelConfig(path_loss_model=PATH_LOSS_MODEL_V2X, shadowing_37885=True,
                             shadow_decorrelation_distance_m=10.0)
    c50 = ChannelModelConfig(path_loss_model=PATH_LOSS_MODEL_V2X, shadowing_37885=True,
                             shadow_decorrelation_distance_m=50.0)
    s10 = evaluate_channel(scene, "veh_0", "veh_1", config=c10).shadowing_db
    s50 = evaluate_channel(scene, "veh_0", "veh_1", config=c50).shadowing_db
    assert s10 != s50


def test_regime_threads_decorr_default_and_explicit() -> None:
    default = PhysicsRegime(tx_power_dbm=20.0, path_loss_model="v2x_37885", shadowing_37885=True)
    assert default.channel_config().shadow_decorrelation_distance_m == SHADOW_DECORRELATION_DISTANCE_M
    swept = PhysicsRegime(tx_power_dbm=20.0, path_loss_model="v2x_37885", shadowing_37885=True,
                          shadow_decorrelation_distance_m=25.0)
    assert swept.channel_config().shadow_decorrelation_distance_m == 25.0
