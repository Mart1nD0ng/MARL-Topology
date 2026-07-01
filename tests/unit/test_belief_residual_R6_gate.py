"""R6 (Belief-Guided Residual PPO) — evidence-gated residual action (heads ACTIVE_IN_EVAL → ACTIVE_IN_DEPLOY).

Load-bearing tests (Contract v4 §3): the gated action CALLS the frozen R5 heads in the decision path and uses
repair/safety/edit to FILTER candidate edits; zero gated candidates → the anchor EXACTLY (effect-on-decision);
a bad-prediction head gates all edits out (→ anchor) while a good-prediction head applies them (topology ≠
anchor, both directions); the result is budget-safe; and the deployed path takes NO evaluator/true-CSI/critic
argument. The deployable A/B "does it beat the anchor?" question is the R6 DECISION (measured by the multi-seed
generator), deliberately NOT asserted here.

Fails on HEAD: `marl_topology.training.evidence_gated_action` does not exist yet.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT / "scripts" / "train", _ROOT / "scripts" / "diagnostics"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _scenes(n=1, frames=2, seed=4):
    from build_operating_point_dataset import operating_point_regime
    from marl_topology.training.dynamic_frames import sample_dynamic_scenes
    from marl_topology.training.two_timescale_env import ReconfigCost
    return sample_dynamic_scenes(seed=seed, count=n, node_count_choices=(8,),
                                 regime=operating_point_regime(20.0), num_frames=frames, dt_s=2.0,
                                 speed_min_mps=15.0, speed_max_mps=30.0, reconfig=ReconfigCost(),
                                 hold_interval=4, gamma=0.95)


def _obs0():
    sc = _scenes(1, 2, seed=4)[0]
    return sc.observation(0, [])


class _StubActor:
    """Edit-score stub: constant per-edge predictions (drives the gate deterministically). Records calls."""
    def __init__(self, edit, repair, safety):
        self.edit, self.repair, self.safety = edit, repair, safety
        self.calls = 0

    def edit_scores(self, nf, ef, ei, hidden=None):
        self.calls += 1
        E = ei.shape[0]
        return ({"edit_logit": torch.full((E,), float(self.edit)),
                 "repair_pred": torch.full((E,), float(self.repair)),
                 "safety_pred": torch.full((E,), float(self.safety))}, None)


def _mean_std(scenes):
    from marl_topology.training.residual_saturation import feature_standardization_all_frames
    return feature_standardization_all_frames(scenes)


def test_gate_calls_heads_in_decision_path() -> None:
    from marl_topology.training.evidence_gated_action import evidence_gated_residual
    obs = _obs0()
    mean, std = _mean_std(_scenes(1, 2, seed=4))
    stub = _StubActor(edit=10.0, repair=10.0, safety=-10.0)        # good edits everywhere
    res = evidence_gated_residual(obs, stub, [], mean, std, tau_edit=0.5, tau_repair=0.0, tau_safety=0.0)
    assert stub.calls >= 1                                          # the heads are CALLED in the decision path
    assert "topology" in res and "anchor" in res and "edit_rate" in res


def test_zero_gated_candidates_returns_anchor() -> None:
    from marl_topology.training.evidence_gated_action import evidence_gated_residual
    obs = _obs0()
    mean, std = _mean_std(_scenes(1, 2, seed=4))
    stub = _StubActor(edit=10.0, repair=10.0, safety=-10.0)
    # tau_edit = 2.0 is unreachable (sigmoid < 1) -> NO candidate passes the gate -> anchor exactly
    res = evidence_gated_residual(obs, stub, [], mean, std, tau_edit=2.0, tau_repair=0.0, tau_safety=0.0)
    assert sorted(res["topology"]) == sorted(res["anchor"])
    assert res["edit_rate"] == 0.0


def test_gate_blocks_bad_edits_and_applies_good() -> None:
    from marl_topology.training.evidence_gated_action import evidence_gated_residual
    obs = _obs0()
    mean, std = _mean_std(_scenes(1, 2, seed=4))
    # BAD head: edit passes (10) but repair_pred=-10 (adds fail repair gate) and safety_pred=+10 (removes fail
    # safety gate) -> NO edit survives -> anchor (proves repair/safety gates are load-bearing, not just edit).
    bad = _StubActor(edit=10.0, repair=-10.0, safety=10.0)
    rb = evidence_gated_residual(obs, bad, [], mean, std, tau_edit=0.5, tau_repair=0.0, tau_safety=0.0)
    assert sorted(rb["topology"]) == sorted(rb["anchor"]) and rb["edit_rate"] == 0.0
    # GOOD head: edit passes, repair high (adds pass), safety low (removes pass) -> edits applied.
    good = _StubActor(edit=10.0, repair=10.0, safety=-10.0)
    rg = evidence_gated_residual(obs, good, [], mean, std, tau_edit=0.5, tau_repair=0.0, tau_safety=0.0)
    assert sorted(rg["topology"]) != sorted(rg["anchor"]) and rg["edit_rate"] > 0.0


def test_gate_is_budget_safe() -> None:
    from marl_topology.budgets import node_budgets_for_scene
    from marl_topology.training.evidence_gated_action import evidence_gated_residual
    obs = _obs0()
    mean, std = _mean_std(_scenes(1, 2, seed=4))
    good = _StubActor(edit=10.0, repair=10.0, safety=-10.0)        # maximally aggressive: add everything
    res = evidence_gated_residual(obs, good, [], mean, std, tau_edit=0.5, tau_repair=0.0, tau_safety=0.0)
    budgets = dict(node_budgets_for_scene(obs["context"].evaluator.scene))
    for node, kept in res["accept"].items():
        assert len(kept) <= int(budgets.get(node, 0))             # no node exceeds its radio budget


def test_gate_deploy_uses_no_evaluator() -> None:
    from marl_topology.training import evidence_gated_action as ega
    # signature audit: the deployed gate takes only the local obs + frozen actor + standardization + thresholds
    params = set(inspect.signature(ega.evidence_gated_residual).parameters)
    for banned in ("evaluator", "ev", "csi", "true_csi", "critic", "oracle", "solver"):
        assert banned not in params
    # the decision-path source must not call the reliability/quorum evaluator to FILTER (those are R5 labels)
    src = (_ROOT / "src" / "marl_topology" / "training" / "evidence_gated_action.py").read_text(encoding="utf-8")
    assert "topology_reliability" not in src and "topology_quorum_deficit" not in src
    assert "edit_scores" in src                                    # the heads ARE the gate signal
