"""Small executable demo for the Stage 2 goal skeleton."""

from __future__ import annotations

from marl_topology.env import MinimalDecPOMDPEnv
from marl_topology.policies import (
    DecentralizedPolicyBaselines,
    build_local_observations,
    run_decentralized_baseline,
)
from marl_topology.scenario import Scene3D, get_scenario_fixture
from marl_topology.topology import CandidateGraph
from marl_topology.topology.evaluator import TopologyEvaluator
from marl_topology.topology.oracle import TopologyOracle

from .fixture_stack import build_fixture_stack


def build_demo_stack() -> tuple[Scene3D, CandidateGraph, TopologyEvaluator, TopologyOracle]:
    stack = build_fixture_stack(get_scenario_fixture("demo_stage2"))
    return stack.scene, stack.graph, stack.evaluator, stack.oracle


def build_demo_env(horizon: int = 1) -> MinimalDecPOMDPEnv:
    scene, graph, evaluator, _ = build_demo_stack()
    return MinimalDecPOMDPEnv(scene=scene, graph=graph, evaluator=evaluator, horizon=horizon)


def build_demo_report() -> dict[str, object]:
    scene, graph, evaluator, oracle = build_demo_stack()
    env = MinimalDecPOMDPEnv(scene=scene, graph=graph, evaluator=evaluator, horizon=1)
    reset = env.reset()
    oracle_result = oracle.solve(random_seed=7)
    baselines = {
        name: {
            "topology_id": evaluation.topology_id,
            "selected_edge_ids": list(evaluation.selected_edge_ids),
            "consensus_success": bool(evaluation.metrics["consensus_success"]),
            "consensus_success_probability": evaluation.metrics["consensus_success_probability"],
            "latency": evaluation.metrics["latency"],
            "energy": evaluation.metrics["energy"],
            "topology_diagnostics": evaluation.metrics["topology_diagnostics"],
        }
        for name, evaluation in oracle_result.baseline_evaluations.items()
    }
    local_observations = build_local_observations(
        scene=scene,
        graph=graph,
        link_records=evaluator.link_records,
        time_step=0,
    )
    decentralized = {
        result.name: {
            "selected_edge_ids": list(result.joint_action.selected_edge_ids),
            "proposal_count": result.joint_action.proposal_count,
            "assembly_rule": result.joint_action.assembly_rule,
        }
        for result in (
            run_decentralized_baseline(
                name="decentralized_no_edges",
                observations=local_observations,
                graph=graph,
                rule=DecentralizedPolicyBaselines.no_edges,
            ),
            run_decentralized_baseline(
                name="decentralized_all_local_edges",
                observations=local_observations,
                graph=graph,
                rule=DecentralizedPolicyBaselines.all_local_edges,
            ),
            run_decentralized_baseline(
                name="decentralized_top1_reliability",
                observations=reset.observations,
                graph=graph,
                rule=lambda obs: DecentralizedPolicyBaselines.top_k_reliability(obs, k=1),
            ),
        )
    }
    return {
        "scenario_id": graph.scenario_id,
        "physics_regime": graph.physics_regime,
        "node_ids": list(graph.node_ids),
        "candidate_edge_ids": list(graph.edge_ids),
        "baselines": baselines,
        "decentralized_baselines": decentralized,
        "env": {
            "reset_time_step": reset.time_step,
            "observation_count": len(reset.observations),
            "horizon": env.horizon,
        },
        "oracle": {
            "status": oracle_result.status,
            "oracle_name": oracle_result.oracle_name,
            "selected_edge_ids": list(oracle_result.selected_edge_ids),
            "searched_topology_count": oracle_result.searched_topology_count,
            "is_exhaustive": oracle_result.is_exhaustive,
        },
    }
