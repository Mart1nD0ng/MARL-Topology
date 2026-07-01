"""T1 (Temporal Recovery) — load-bearing tests for the CSI-recovery upper-bound oracle (THE GATE).

The oracle feeds the local_hysteresis anchor four different psucc vectors (stale / direct-neural /
physics-residual / true) and scores the produced topology with the TRUE evaluator. These tests pin the three
properties the gate's validity rests on (Contract v4 §3/§4):
  * the recovered psucc is genuinely INJECTED into the anchor decision (not a no-op),
  * the direct/physics predictors' INFERENCE is LEAK-FREE (never reads the true current CSI; the true CSI is a
    training-only label), and
  * the floor arm reads stale psucc and the ceiling arm reads the true current psucc.
Fails on HEAD: the T1 generator does not exist yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT / "scripts" / "train", _ROOT / "scripts" / "diagnostics"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


class _A:
    dyn_nodes = [8]
    frames = 4
    gamma = 0.95


def _stale_scene(seed: int = 3):
    import t1_oracle_recovery_gen as g
    from marl_topology.training.csi_observation_model import CsiObservationModel
    return g.build_csi_scenes("random", seed, 1, _A(), CsiObservationModel(mode="delay", delay_frames=1))[0]


def test_recovered_psucc_injects_into_anchor() -> None:
    """The anchor decides purely on psucc col 0; the oracle overwrites it. Identity when psucc==stale, and a
    high vs low psucc vector yields a strictly different (larger) topology -> the injection is load-bearing."""
    import t1_oracle_recovery_gen as g
    from marl_topology.training.edit_head_training import _anchor

    sc = _stale_scene()
    obs = sc.observation(1, [])
    _feats, stale = g._edge_features(sc, 1, obs)

    # (a) identity: injecting the stale psucc reproduces the plain anchor exactly.
    inj = g._anchor_with_psucc(obs, [], stale)
    base = _anchor(obs, [])
    assert set(inj) == set(base)

    # (b) load-bearing: a high psucc admits edges the anchor keeps/adds; an all-zero psucc admits none.
    e = obs["ef"].shape[0]
    hi = g._anchor_with_psucc(obs, [], torch.full((e,), 0.99))
    lo = g._anchor_with_psucc(obs, [], torch.zeros(e))
    assert len(hi) > len(lo)


def test_predictor_inference_is_leak_free(monkeypatch) -> None:
    """Feature extraction + predictor training + inference NEVER read the true current CSI; only the LABEL path
    (_true_psucc -> belief_target_logits) does. Spy the true-CSI reader and assert 0 calls through inference."""
    import t1_oracle_recovery_gen as g

    calls = {"n": 0}
    orig = g.belief_target_logits

    def _spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(g, "belief_target_logits", _spy)

    sc = _stale_scene()
    obs = sc.observation(1, [])
    feats, stale = g._edge_features(sc, 1, obs)          # inference features (leak-free)
    y = torch.rand(feats.shape[0])                        # dummy label (real labels are training-only)
    net, mu, sd = g._train_predictor(feats, y, stale, "physics", seed=0, epochs=5)
    _pred = g._predict(net, mu, sd, feats, stale, "physics")
    assert calls["n"] == 0                                # inference NEVER read the true current CSI

    _true = g._true_psucc(sc, 1, obs["edge_ids"])         # the LABEL path (training-only) DOES read it
    assert calls["n"] == 1


def test_floor_is_stale_ceiling_is_true() -> None:
    """Arm A (floor) reads the stale observed psucc; arm D (ceiling) reads the true current psucc."""
    import t1_oracle_recovery_gen as g
    from marl_topology.training.csi_belief import belief_target_logits

    sc = _stale_scene()
    obs = sc.observation(1, [])
    _feats, stale = g._edge_features(sc, 1, obs)
    assert torch.allclose(stale, obs["ef"][:, 0])                        # floor == stale col 0

    true = g._true_psucc(sc, 1, obs["edge_ids"])
    exp = torch.sigmoid(belief_target_logits(sc, 1, obs["edge_ids"]))    # ceiling == true current channel
    assert torch.allclose(true, exp)
