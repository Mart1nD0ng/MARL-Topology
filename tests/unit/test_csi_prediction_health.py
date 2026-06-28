"""Q2 (POMDP-QP-FAR): CSI-prediction health-check diagnostic.

Pins the testable core of scripts/diagnostics/csi_prediction_health.py: the extraction correctly pairs
the OBSERVED (stale) channel with the TRUE current target (no leak), the identity baseline is exactly 0
under current CSI and positive under delay, and a velocity-aware predictor can beat identity when the
motion feature carries the channel trend (the mechanism premise). Fails on HEAD (module is new).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts" / "diagnostics"))

from marl_topology.data.stage31_scenario_generator import PhysicsRegime, measure_reliable_range_m
from marl_topology.geometry3d import Point3D
from marl_topology.scenario.scene import Node3D, NodeKind, NodeMotion, Scene3D
from marl_topology.training.csi_observation_model import CsiObservationModel
from marl_topology.training.dynamic_frames import dynamic_scene_from_motion
from marl_topology.training.two_timescale_env import ReconfigCost

import csi_prediction_health as cph  # noqa: E402

_R = measure_reliable_range_m(PhysicsRegime())


def _scene(csi, *, num_frames=5):
    nodes = (
        Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 5.0)),
        Node3D("veh_0", NodeKind.VEHICLE, Point3D(0.55 * _R, 0.0, 1.5)),
        Node3D("veh_1", NodeKind.VEHICLE, Point3D(0.0, 0.2 * _R, 1.5)),
    )
    base = Scene3D(scenario_id="q2", nodes=nodes)
    return dynamic_scene_from_motion(
        base, (NodeMotion("veh_0", (0.2 * _R, 0.0, 0.0)),), PhysicsRegime(), quorum_size=3,
        num_frames=num_frames, dt_s=1.0, reliable_range_m=_R, reconfig=ReconfigCost(),
        csi_observation_model=csi)


def test_extract_pairs_observed_history_with_true_current() -> None:
    scene = _scene(CsiObservationModel(mode="delay", delay_frames=1))
    series = cph.extract_series([scene], CsiObservationModel(mode="delay", delay_frames=1))
    assert series and all(len(s) == scene.n_frames for s in series)
    for seq in series:
        # obs_psucc, age, rel_vel, dist_delta, true_psucc
        assert seq[0][0] == seq[0][4]            # frame 0: observed == true current (clamp)
        assert seq[0][1] == 0.0                  # age 0 at frame 0
        for t in range(1, len(seq)):
            assert abs(seq[t][0] - seq[t - 1][4]) < 1e-9   # delay-1 observed(t) == true(t-1)
            assert seq[t][1] == 1.0              # age = 1 every later frame (delay-1, always probed)


def test_identity_mse_zero_under_current() -> None:
    scene = _scene(CsiObservationModel(mode="current"))
    series = cph.extract_series([scene], CsiObservationModel(mode="current"))
    ident = cph.identity_metrics(series)
    assert ident["mse"] == 0.0                   # observed == true everywhere -> exact 0


def test_identity_mse_positive_under_delay() -> None:
    scene = _scene(CsiObservationModel(mode="delay", delay_frames=1))
    series = cph.extract_series([scene], CsiObservationModel(mode="delay", delay_frames=1))
    ident = cph.identity_metrics(series)
    assert ident["mse"] > 1e-6                   # the moving vehicle's link changes -> stale != current


def test_velocity_linear_beats_identity_on_monotone_channel() -> None:
    # Synthetic premise test: true psucc ramps; observed lags by 1; rel_vel carries the constant slope.
    # A velocity-aware predictor can recover true = obs + slope -> far below identity's constant lag error.
    series = []
    for i in range(24):
        base = 0.30 + 0.004 * i
        seq = []
        for t in range(6):
            true_p = base + 0.05 * t
            obs_p = true_p if t == 0 else (base + 0.05 * (t - 1))   # lag-1
            seq.append((obs_p, float(min(t, 1)), 0.05, 0.05, true_p))   # rel_vel encodes the slope
        series.append(seq)
    ident = cph.identity_metrics(series)
    mem_vel = cph.train_eval(series, series, arch="memoryless", use_velocity=True, epochs=600, seed=0)
    assert ident["mse"] > 1e-4
    assert mem_vel["mse"] < 0.5 * ident["mse"]   # velocity lets the predictor undo the lag


def test_spearman_perfect_on_monotone() -> None:
    pred = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    true = torch.tensor([[0.5, 0.6, 0.7, 0.8]])
    mask = torch.ones_like(pred)
    assert cph._spearman(pred, true, mask) > 0.99   # rank-monotone -> Spearman ~1
