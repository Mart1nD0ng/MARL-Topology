"""R8 (Belief-Guided Residual PPO) — the stale-CSI PREMISE test (the campaign's actual failure regime).

R1-R7 ran mostly with the stale-CSI overlay OFF (current-frame channel). R8 turns it ON
(CsiObservationModel(mode='delay', delay_frames=1)): the deployed actor observes a STALE channel while the
evaluator/reward stay on the TRUE current channel. The question: does the evidence-gated action (heads retrained
on STALE features, labels from the TRUE current evaluator = training-only) REPAIR the stale feasibility drop?

Load-bearing tests (Contract v4 §3): the stale overlay is ACTIVE in the deployed observation AND the evaluator
stays on the true channel (leak-free -- the actor sees stale, the score is true). Fails on HEAD: the R8 stale
build helper does not exist yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT / "scripts" / "train", _ROOT / "scripts" / "diagnostics"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def test_stale_overlay_changes_obs_not_evaluator() -> None:
    """delay-1 stale CSI: the observed edge features DIFFER from the current channel at t>=1, but the evaluator
    (true PBFT on the current channel) is UNCHANGED by the overlay -- the actor sees stale, the score is true."""
    import r8_stale_csi_gen as g
    from marl_topology.training.csi_observation_model import CsiObservationModel
    from marl_topology.training.quorum_deficit_bridge import topology_reliability

    class _A:
        dyn_nodes = [8]
        frames = 4
        gamma = 0.95

    stale = g.build_csi_scenes("random", 7, 1, _A(), CsiObservationModel(mode="delay", delay_frames=1))[0]
    curr = g.build_csi_scenes("random", 7, 1, _A(), None)[0]
    # the stale overlay staleifies the CSI cols (0-3) and appends [csi_age, csi_observed_mask] (dim +2), so
    # the OBSERVED CSI columns diverge from the current channel at some t>=1 (the overlay is active)
    diffs = 0
    for t in range(1, stale.n_frames):
        es = stale.observation(t, [])["ef"]
        ec = curr.observation(t, [])["ef"]
        base = min(es.shape[1], ec.shape[1])                      # shared base cols (stale appends 2 extra)
        if float((es[:, :4] - ec[:, :4]).abs().max()) > 1e-6:     # CSI cols 0-3 staleified vs current
            diffs += 1
        assert es.shape[1] == ec.shape[1] + 2                     # stale appends [csi_age, csi_observed_mask]
    assert diffs >= 1, "delay-1 stale overlay must change the observed CSI columns at some frame t>=1"
    # the EVALUATOR (true current channel) is identical between the stale and current builds -> leak-free score
    for t in range(stale.n_frames):
        obs_s = stale.observation(t, [])
        obs_c = curr.observation(t, [])
        topo = list(obs_c["edge_ids"])[:3]
        rs = topology_reliability(obs_s["context"].evaluator, topo)["consensus"]
        rc = topology_reliability(obs_c["context"].evaluator, topo)["consensus"]
        assert abs(float(rs) - float(rc)) < 1e-9, "the evaluator must stay on the TRUE current channel (no leak)"


def test_build_csi_scenes_active_flag() -> None:
    import r8_stale_csi_gen as g
    from marl_topology.training.csi_observation_model import CsiObservationModel

    class _A:
        dyn_nodes = [8]
        frames = 3
        gamma = 0.95

    stale = g.build_csi_scenes("random", 1, 1, _A(), CsiObservationModel(mode="delay", delay_frames=1))
    curr = g.build_csi_scenes("random", 1, 1, _A(), None)
    assert len(stale) == 1 and len(curr) == 1
    # the deployed decode helper is the SAME evidence-gated action (0 eval) used at R6
    src = (_ROOT / "scripts" / "diagnostics" / "r8_stale_csi_gen.py").read_text(encoding="utf-8")
    assert "evidence_gated_residual" in src and "topology_reliability" in src
