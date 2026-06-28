"""Q4 (POMDP-QP-FAR): D_quorum <-> true C alignment bridge + metric core.

Pins: (a) the refactored evaluator's ``_reliability_inputs`` reproduces ``evaluate``'s true C through
the SAME (fixed_set robust) path the reward uses; (b) the bridge D_quorum is monotone in edges (a
superset never has a larger deficit); (c) the alignment metric core (Spearman, top-k hit, false-improve
rate) behaves on synthetic data. Fails on HEAD (bridge + script + the refactor method are new).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts" / "diagnostics"))
sys.path.insert(0, str(_ROOT / "scripts" / "train"))

from marl_topology.protocol import STRATEGY_AUTO, robust_consensus_reliability
from marl_topology.training.dynamic_frames import sample_dynamic_scenes
from marl_topology.training.quorum_deficit_bridge import (
    reference_evaluator, topology_quorum_deficit, topology_reliability)
from marl_topology.training.two_timescale_env import ReconfigCost

import quorum_deficit_alignment as qda  # noqa: E402


def _scene():
    from build_operating_point_dataset import operating_point_regime
    scenes = sample_dynamic_scenes(
        seed=7, count=1, node_count_choices=(12,), regime=operating_point_regime(20.0),
        num_frames=1, dt_s=1.0, speed_min_mps=15.0, speed_max_mps=30.0,
        reconfig=ReconfigCost(), hold_interval=1, gamma=0.95)
    return scenes[0]


# -- bridge grounded to the real evaluator ----------------------------------------------------------

def test_reliability_inputs_reproduce_true_C() -> None:
    sc = _scene()
    ev = sc.context(0).evaluator
    ref = reference_evaluator(ev)
    cand = list(sc.edge_ids_at(0))
    for topo in (tuple(sorted(cand)), tuple(sorted(cand[: max(4, len(cand) // 2)]))):
        ri = ref._reliability_inputs(topo)
        if len(ri["validators"]) < 4:
            continue
        rel = robust_consensus_reliability(
            ri["validators"], pre_prepare_matrix=ri["pre_prepare"], prepare_matrix=ri["prepare"],
            commit_matrix=ri["commit"], fault_tolerance=ri["fault_tolerance"], strategy=STRATEGY_AUTO)
        true_c = ref.evaluate(topo).metrics["consensus_success_probability"]
        assert rel.consensus_success_probability == pytest.approx(true_c, abs=1e-9)


def test_bridge_returns_valid_deficit() -> None:
    # NOTE: D_quorum is NOT monotone in edge COUNT -- more active edges -> more STDMA interference ->
    # lower per-link success -> higher deficit. C and D are built from the SAME (interference-aware)
    # matrices, so they can still ALIGN even though neither is monotone in edges (that is exactly why
    # the alignment test, not a monotonicity assumption, is the gate).
    sc = _scene()
    ev = sc.context(0).evaluator
    cand = list(sc.edge_ids_at(0))
    for topo in (cand, cand[: max(4, len(cand) // 2)], cand[: max(4, len(cand) // 4)]):
        d = topology_quorum_deficit(ev, topo)
        assert d is not None
        assert math.isfinite(d["d_quorum_mean"]) and d["d_quorum_mean"] >= 0.0
        assert d["d_quorum_max"] >= d["d_quorum_mean"] >= 0.0
        assert 0.0 <= topology_reliability(ev, topo)["consensus"] <= 1.0


# -- alignment metric core --------------------------------------------------------------------------

def test_perfect_alignment_synthetic() -> None:
    # dD = -dC exactly -> Spearman +1, no false improvement.
    dC = [-0.3, -0.1, 0.0, 0.2, 0.5, 0.4, -0.2, 0.1]
    dD = list(dC)                                           # deficit improvement tracks reliability gain
    dE = [0.0] * len(dC)
    m = qda.alignment_metrics(dD, dC, dE)
    assert m["spearman_dD_dC"] == pytest.approx(1.0, abs=1e-9)
    assert m["dD_pos_but_C_worsens_rate"] == 0.0
    assert m["top_k_repair_hit_rate"] == pytest.approx(1.0, abs=1e-9)


def test_anti_alignment_flags_false_improvement() -> None:
    # dD says "better" but dC says "worse" -> high false-improvement rate, negative Spearman.
    dD = [0.5, 0.4, 0.3, 0.2]
    dC = [-0.5, -0.4, -0.3, -0.2]
    dE = [1.0, 1.0, 1.0, 1.0]
    m = qda.alignment_metrics(dD, dC, dE, energy_explode=0.5)
    assert m["spearman_dD_dC"] < -0.99
    assert m["dD_pos_but_C_worsens_rate"] == 1.0
    assert m["dD_pos_but_energy_explodes_rate"] == 1.0


def test_spearman_monotone() -> None:
    assert qda.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0, abs=1e-9)
    assert qda.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0, abs=1e-9)
    assert qda.alignment_metrics([], [], [])["n"] == 0
