"""Q1 (POMDP-QP-FAR): stale / partial CSI observation model.

The dynamic actor decides on an OBSERVED (lagged / partially-probed) channel while the reward /
evaluator keep the TRUE current channel (Spec S4.6, Contract D1). These tests pin: (a) the pure model
math (deterministic probe mask, source/age formula, hold-last), (b) the scene integration (current
byte-identical; delay uses the previous frame; age increments), and (c) the NO-LEAK invariant -- the
evaluator (the reward source) is unaffected by staleifying the actor observation.

All of these fail on HEAD: ``CsiObservationModel`` and the ``csi_observation_model`` scene parameter
did not exist before Q1.
"""

from __future__ import annotations

import torch

from marl_topology.data.stage31_scenario_generator import PhysicsRegime, measure_reliable_range_m
from marl_topology.geometry3d import Point3D
from marl_topology.scenario.scene import Node3D, NodeKind, NodeMotion, Scene3D
from marl_topology.training.csi_observation_model import CsiObservationModel
from marl_topology.training.dynamic_frames import dynamic_scene_from_motion
from marl_topology.training.two_timescale_env import ReconfigCost

_R = measure_reliable_range_m(PhysicsRegime())


def _scene(motions, csi, *, num_frames=4):
    # A moving vehicle near the range boundary -> its incident link CSI EVOLVES frame-to-frame, so
    # "stale != current" is observable. RSU + 2 vehicles -> 3 nodes, 3 candidate edges.
    nodes = (
        Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 5.0)),
        Node3D("veh_0", NodeKind.VEHICLE, Point3D(0.55 * _R, 0.0, 1.5)),
        Node3D("veh_1", NodeKind.VEHICLE, Point3D(0.0, 0.2 * _R, 1.5)),
    )
    base = Scene3D(scenario_id="csi", nodes=nodes)
    return dynamic_scene_from_motion(
        base, motions, PhysicsRegime(), quorum_size=3, num_frames=num_frames, dt_s=1.0,
        reliable_range_m=_R, reconfig=ReconfigCost(e_edge=0.05, l_edge=0.0),
        csi_observation_model=csi)


# veh_0 drives outward, crossing the reliable-range boundary over the window -> the CSI block (psucc /
# latency / energy) genuinely evolves frame-to-frame.
_MOTION = (NodeMotion("veh_0", (0.2 * _R, 0.0, 0.0)),)
_CSI = slice(0, 4)   # the staleified channel cols {psucc, psucc, latency, energy}


# -- pure model math --------------------------------------------------------------------------------

def test_csi_model_probe_mask_deterministic() -> None:
    m = CsiObservationModel(mode="partial", probe_probability=0.5, seed=3)
    assert m.probe("a--b", 0) is True                       # frame 0 is always the initial probe
    assert m.probe("a--b", 5) == m.probe("a--b", 5)         # reproducible (stable hash, not process hash)
    always = CsiObservationModel(mode="delay", delay_frames=1)
    assert all(always.probe("a--b", t) for t in range(6))   # delay mode probes every frame
    never = CsiObservationModel(mode="partial", probe_probability=0.0)
    assert never.probe("a--b", 0) is True and not never.probe("a--b", 1)


def test_csi_model_source_age_formula() -> None:
    cur = CsiObservationModel(mode="current")
    assert not cur.is_active()
    assert cur.edge_plan("e", 3) == (3, 0, True)            # current -> no staleness
    d = CsiObservationModel(mode="delay", delay_frames=2)
    assert d.is_active()
    assert d.edge_plan("e", 0) == (0, 0, True)              # t<delta clamps to frame 0
    assert d.edge_plan("e", 1) == (0, 1, True)
    assert d.edge_plan("e", 5) == (3, 2, True)              # src=t-delta, age=delta


def test_partial_holds_last_and_age_increments_model() -> None:
    m = CsiObservationModel(mode="partial", probe_probability=0.5, seed=7)
    eid = "n3--n9"
    last = 0
    for t in range(8):
        src, age, now = m.edge_plan(eid, t)
        assert now == m.probe(eid, t)                       # observed_now matches the probe mask
        if m.probe(eid, t):
            last = t
        assert src == last and age == t - last              # hold-last value; age = frames since probe
    assert any(m.edge_plan(eid, t)[1] > 0 for t in range(12))   # the mechanism is non-trivial (some staleness)


# -- scene integration ------------------------------------------------------------------------------

def test_current_mode_byte_identical() -> None:
    none = _scene(_MOTION, None)
    cur = _scene(_MOTION, CsiObservationModel(mode="current"))
    for t in range(none.n_frames):
        ef_none = none.observation(t, [])["ef"]
        ef_cur = cur.observation(t, [])["ef"]
        assert ef_cur.shape == ef_none.shape                # no appended columns when inactive
        assert torch.equal(ef_cur, ef_none)                 # byte-identical


def test_delay1_observation_uses_previous_frame_csi() -> None:
    none = _scene(_MOTION, None)
    delay = _scene(_MOTION, CsiObservationModel(mode="delay", delay_frames=1))
    assert delay.observation(0, [])["ef"].shape[1] == none.observation(0, [])["ef"].shape[1] + 2  # +[age,mask]
    # frame 0: observed == true current (clamp)
    assert torch.allclose(delay.observation(0, [])["ef"][:, _CSI], none.observation(0, [])["ef"][:, _CSI])
    for t in range(1, none.n_frames):
        observed_now = delay.observation(t, [])["ef"][:, _CSI]   # channel CSI the actor sees at t
        true_prev = none.observation(t - 1, [])["ef"][:, _CSI]   # true channel at t-1
        assert torch.allclose(observed_now, true_prev)           # delay-1 shows the previous frame


def test_partial_csi_holds_last_observation() -> None:
    none = _scene(_MOTION, None)
    hold = _scene(_MOTION, CsiObservationModel(mode="partial", probe_probability=0.0))  # only frame 0 probed
    held_psucc = none.observation(0, [])["ef"][:, 0]
    for t in range(1, none.n_frames):
        assert torch.allclose(hold.observation(t, [])["ef"][:, 0], held_psucc)          # held at frame-0 value


def test_csi_age_increments_when_unobserved() -> None:
    hold = _scene(_MOTION, CsiObservationModel(mode="partial", probe_probability=0.0))
    for t in range(hold.n_frames):
        ef = hold.observation(t, [])["ef"]
        age_col, mask_col = ef[:, -2], ef[:, -1]            # appended [csi_age, csi_observed_mask]
        assert torch.allclose(age_col, torch.full_like(age_col, float(t)))     # age == frames since probe
        assert torch.allclose(mask_col, torch.full_like(mask_col, 1.0 if t == 0 else 0.0))


def test_reward_uses_true_current_csi_not_observed_csi() -> None:
    # reward_of -> _evaluate reads sample["context"].evaluator (train_decentralized_rl.py L130), NOT ef.
    # So staleifying the actor observation must NOT change the evaluator metrics (the reward source).
    none = _scene(_MOTION, None)
    delay = _scene(_MOTION, CsiObservationModel(mode="delay", delay_frames=2))
    topo = set(none.observation(0, [])["edge_ids"])
    for t in range(none.n_frames):
        m_true = none.observation(t, [])["context"].evaluator.evaluate(topo).metrics
        m_stale = delay.observation(t, [])["context"].evaluator.evaluate(topo).metrics
        assert m_stale["consensus_success_probability"] == m_true["consensus_success_probability"]
        assert m_stale.get("energy", 0.0) == m_true.get("energy", 0.0)


def test_actor_never_receives_true_current_csi_under_stale_mode() -> None:
    none = _scene(_MOTION, None)
    delay = _scene(_MOTION, CsiObservationModel(mode="delay", delay_frames=1))
    differs = False
    for t in range(1, none.n_frames):
        observed_now = delay.observation(t, [])["ef"][:, _CSI]
        true_now = none.observation(t, [])["ef"][:, _CSI]   # the TRUE current channel the actor must NOT see
        if not torch.allclose(observed_now, true_now):
            differs = True
    assert differs, "stale CSI must differ from the true current channel on a moving link"
