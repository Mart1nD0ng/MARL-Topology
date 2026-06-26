"""D7: fair deployable baselines (local-only action; zero evaluator calls) vs central references
(action selection calls the evaluator -- clearly marked central). Contract v3 §10.1/§10.2.
"""

from __future__ import annotations

import torch

from marl_topology.data.stage31_scenario_generator import (
    PhysicsRegime,
    measure_reliable_range_m,
)
from marl_topology.geometry3d import Point3D
from marl_topology.scenario.scene import Node3D, NodeKind, NodeMotion, Scene3D
from marl_topology.training.dynamic_frames import dynamic_scene_from_motion
from marl_topology.training.dynamic_rl import _budgets_edges
from marl_topology.training.two_timescale_env import ReconfigCost

_R = measure_reliable_range_m(PhysicsRegime())


def _scene(num_frames=3):
    base = Scene3D(scenario_id="d7", nodes=(
        Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 5.0)),
        Node3D("veh_0", NodeKind.VEHICLE, Point3D(0.2 * _R, 0.0, 1.5)),
        Node3D("veh_1", NodeKind.VEHICLE, Point3D(0.0, 0.2 * _R, 1.5)),
        Node3D("veh_2", NodeKind.VEHICLE, Point3D(0.2 * _R, 0.2 * _R, 1.5)),
    ))
    motions = (NodeMotion("veh_2", (0.3 * _R, 0.0, 0.0)),)
    return dynamic_scene_from_motion(base, motions, PhysicsRegime(), quorum_size=3, num_frames=num_frames,
                                     dt_s=1.0, reliable_range_m=_R, reconfig=ReconfigCost(e_edge=0.05, l_edge=0.0))


def _reward_fn(obs, edges, e_ref, lam_c, lam_b, beta, reward_mode):
    c = float(obs["context"].evaluator.evaluate(set(edges)).metrics["consensus_success_probability"])
    return (c - 0.9), max(0.0, 0.9 - c), 0.0, c >= 0.9


def _ref_energy(obs):
    return 1.0


_RW = dict(reward_of=_reward_fn, ref_energy=_ref_energy, lam_c=1.0, lam_b=1.0, beta=0.1, reward_mode="dense")


def test_deployable_baseline_does_not_call_evaluator_for_action() -> None:
    from marl_topology.training.dynamic_baselines import (
        evaluate_deployable_baseline,
        local_threshold_action,
    )
    scene = _scene()
    res = evaluate_deployable_baseline(
        lambda ef, eids, edges, bud, prev: local_threshold_action(ef, eids, edges, bud, prev, threshold=0.5),
        [scene], label="local_threshold", **_RW)
    assert res["action_evaluator_calls"] == 0          # the ACTION used no evaluator call
    assert res["group"] == "deployable_policy"
    assert 0.0 <= res["per_frame_feasibility"] <= 1.0
    assert "mean_episode_return" in res and "mean_switches_per_frame" in res
    # the action fn is a PURE function of local features -- it takes no context / evaluator argument.
    obs = scene.observation(0, [])
    budgets, edges = _budgets_edges(obs["context"])
    topo = local_threshold_action(obs["ef"], obs["edge_ids"], edges, budgets, [], threshold=0.5)
    assert isinstance(topo, list)
    for node, b in budgets.items():                    # mutual-acceptance respects budgets
        deg = sum(1 for e in topo if node in edges[e])
        assert deg <= b


def test_reference_uses_evaluator_and_is_marked_central() -> None:
    from marl_topology.training.dynamic_baselines import evaluate_central_reference
    scene = _scene()
    res = evaluate_central_reference([scene], label="myopic_greedy", **_RW)
    assert res["group"] == "central_reference"
    assert res["action_evaluator_calls"] > 0           # candidate search USES the evaluator (not deployable)
    assert 0.0 <= res["per_frame_feasibility"] <= 1.0


def test_baseline_budget_report_records_eval_calls() -> None:
    from marl_topology.training.dynamic_baselines import (
        baseline_budget_report,
        evaluate_central_reference,
        evaluate_deployable_baseline,
        local_hysteresis_action,
    )
    scene = _scene()
    dep = evaluate_deployable_baseline(
        lambda ef, eids, edges, bud, prev: local_hysteresis_action(ef, eids, edges, bud, prev,
                                                                   keep_threshold=0.4, add_threshold=0.6),
        [scene], label="local_hysteresis", **_RW)
    cen = evaluate_central_reference([scene], label="myopic_greedy", **_RW)
    report = baseline_budget_report([dep, cen])
    assert set(report["groups"]) == {"deployable_policy", "central_reference"}
    dep_entry = report["groups"]["deployable_policy"][0]
    cen_entry = report["groups"]["central_reference"][0]
    assert dep_entry["action_evaluator_calls"] == 0
    assert cen_entry["action_evaluator_calls"] > 0
    assert report["deployable_use_no_evaluator_for_action"] is True
