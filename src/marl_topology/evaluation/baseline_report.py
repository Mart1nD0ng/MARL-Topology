"""Stage 2.4 deterministic baseline evaluation report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from marl_topology.env import (
    ActorObservation,
    EdgeActionDecision,
    MinimalDecPOMDPEnv,
    ensure_actor_observations_do_not_contain_metrics,
)
from marl_topology.metrics import REGISTERED_METRICS, require_registered_metrics
from marl_topology.policies import DecentralizedPolicyBaselines
from marl_topology.scenario import get_scenario_fixture, iter_scenario_fixtures
from marl_topology.topology.evaluator import TopologyEvaluation

from .fixture_stack import build_fixture_stack


LocalDecisionRule = Callable[[ActorObservation], tuple[EdgeActionDecision, ...]]


@dataclass(frozen=True, slots=True)
class BaselineReportRow:
    name: str
    family: str
    decision_source: str
    topology_id: str
    selected_edge_ids: tuple[str, ...]
    is_oracle: bool
    metrics: Mapping[str, object]
    metric_rows: tuple[Mapping[str, object], ...]
    decision_diagnostics: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "family": self.family,
            "decision_source": self.decision_source,
            "topology_id": self.topology_id,
            "selected_edge_ids": list(self.selected_edge_ids),
            "is_oracle": self.is_oracle,
            "metrics": dict(self.metrics),
            "metric_rows": [dict(row) for row in self.metric_rows],
            "decision_diagnostics": dict(self.decision_diagnostics),
        }


def build_stage2_baseline_evaluation_report(
    fixture_id: str = "demo_stage2",
    random_seed: int = 7,
    threshold_min_success_probability: float = 0.65,
    local_random_seed: int = 13,
    local_random_edge_probability: float = 0.4,
    top_k: int = 1,
) -> dict[str, object]:
    """Build a reproducible baseline report over one deterministic fixture.

    This is an evaluation sensor only. It does not compute rewards, train
    policies, create replay data, or use oracle labels in actor observations.
    """

    if not 0.0 <= threshold_min_success_probability <= 1.0:
        raise ValueError("threshold_min_success_probability must be in [0, 1]")
    if not 0.0 <= local_random_edge_probability <= 1.0:
        raise ValueError("local_random_edge_probability must be in [0, 1]")
    if top_k < 0:
        raise ValueError("top_k must be nonnegative")

    stack = build_fixture_stack(get_scenario_fixture(fixture_id))
    scene = stack.scene
    graph = stack.graph
    evaluator = stack.evaluator
    oracle = stack.oracle
    oracle_result = oracle.solve(random_seed=random_seed)

    baseline_rows: list[BaselineReportRow] = []
    for name, evaluation in oracle_result.baseline_evaluations.items():
        baseline_rows.append(
            _row_from_evaluation(
                name=name,
                family="global_non_learning",
                decision_source="candidate_graph_baseline",
                evaluation=evaluation,
                is_oracle=False,
                decision_diagnostics={
                    "uses_actor_observation": False,
                    "is_full_graph_baseline": bool(
                        evaluation.diagnostics["is_full_graph_baseline"]
                    ),
                },
            )
        )

    decentralized_specs: tuple[tuple[str, LocalDecisionRule, Mapping[str, object]], ...] = (
        (
            "decentralized_no_edges",
            DecentralizedPolicyBaselines.no_edges,
            {"rule": "no_edges"},
        ),
        (
            "decentralized_all_local_edges",
            DecentralizedPolicyBaselines.all_local_edges,
            {"rule": "all_local_edges"},
        ),
        (
            f"decentralized_top{top_k}_reliability",
            lambda obs: DecentralizedPolicyBaselines.top_k_reliability(obs, k=top_k),
            {"rule": "top_k_reliability", "k": top_k},
        ),
        (
            f"decentralized_threshold_{_format_probability_tag(threshold_min_success_probability)}",
            lambda obs: DecentralizedPolicyBaselines.reliability_threshold(
                obs,
                min_success_probability=threshold_min_success_probability,
            ),
            {
                "rule": "reliability_threshold",
                "min_success_probability": threshold_min_success_probability,
            },
        ),
        (
            (
                f"decentralized_random_seed_{local_random_seed}_"
                f"{_format_probability_tag(local_random_edge_probability)}"
            ),
            lambda obs: DecentralizedPolicyBaselines.local_random(
                obs,
                seed=local_random_seed,
                edge_probability=local_random_edge_probability,
            ),
            {
                "rule": "local_random",
                "seed": local_random_seed,
                "edge_probability": local_random_edge_probability,
            },
        ),
    )

    for name, rule, parameters in decentralized_specs:
        env = MinimalDecPOMDPEnv(scene=scene, graph=graph, evaluator=evaluator, horizon=1)
        reset = env.reset()
        ensure_actor_observations_do_not_contain_metrics(reset.observations)
        decisions = tuple(decision for obs in reset.observations for decision in rule(obs))
        step = env.step(decisions, topology_id=f"baseline:{name}")
        ensure_actor_observations_do_not_contain_metrics(step.observations)
        baseline_rows.append(
            _row_from_evaluation(
                name=name,
                family="decentralized_non_learning",
                decision_source="actor_observation_to_edge_action_decision",
                evaluation=step.evaluation,
                is_oracle=False,
                decision_diagnostics={
                    "uses_actor_observation": True,
                    "actor_observation_count": len(reset.observations),
                    "proposal_count": step.joint_action.proposal_count,
                    "assembly_rule": step.joint_action.assembly_rule,
                    "step_time": step.time_step,
                    "terminated": step.terminated,
                    "parameters": dict(parameters),
                    "is_full_graph_baseline": bool(
                        step.evaluation.diagnostics["is_full_graph_baseline"]
                    ),
                },
            )
        )

    oracle_payload = _oracle_payload(oracle_result)
    return {
        "stage": "stage_2_4_baseline_evaluation_report",
        "fixture_id": stack.fixture.fixture_id,
        "fixture_description": stack.fixture.description,
        "expected_oracle_status": stack.fixture.expected_oracle_status,
        "scenario": {
            "scenario_id": graph.scenario_id,
            "physics_regime": graph.physics_regime,
            "node_ids": list(graph.node_ids),
            "candidate_edge_ids": list(graph.edge_ids),
            "candidate_edge_count": len(graph.edge_ids),
            "protocol_variant": evaluator.consensus_config.protocol_variant,
            "quorum_size": evaluator.consensus_config.quorum_size,
            "deadline_s": evaluator.consensus_config.deadline_s,
        },
        "metric_governance": {
            "registered_metric_names": list(REGISTERED_METRICS.keys()),
            "metric_rows_are_registered": all(
                set(row.metrics) <= set(REGISTERED_METRICS) for row in baseline_rows
            ),
            "new_metric_names_introduced": [],
        },
        "baseline_rows": [row.to_payload() for row in baseline_rows],
        "oracle_reference": oracle_payload,
        "checks": {
            "expected_oracle_status_matches": (
                oracle_result.status == stack.fixture.expected_oracle_status
            ),
            "expected_full_graph_success_matches": _full_graph_success_matches(
                baseline_rows,
                expected=stack.fixture.expected_full_graph_success,
            ),
            "full_graph_is_baseline_not_oracle": _full_graph_is_not_oracle(
                baseline_rows,
                oracle_payload,
            ),
            "all_decentralized_rows_use_actor_observation": all(
                row.decision_diagnostics.get("uses_actor_observation") is True
                for row in baseline_rows
                if row.family == "decentralized_non_learning"
            ),
            "no_actor_observation_metric_leakage_detected": True,
            "training_run": False,
            "v5_code_migrated": False,
        },
    }


def build_stage2_scenario_fixture_report() -> dict[str, object]:
    scenario_reports = [
        build_stage2_baseline_evaluation_report(fixture_id=fixture.fixture_id)
        for fixture in iter_scenario_fixtures()
    ]
    return {
        "stage": "stage_2_5_scenario_fixture_contract",
        "fixture_ids": [report["fixture_id"] for report in scenario_reports],
        "scenario_reports": scenario_reports,
        "checks": {
            "fixture_ids_are_unique": len({report["fixture_id"] for report in scenario_reports})
            == len(scenario_reports),
            "all_expected_oracle_statuses_match": all(
                report["checks"]["expected_oracle_status_matches"]
                for report in scenario_reports
            ),
            "all_expected_full_graph_success_values_match": all(
                report["checks"]["expected_full_graph_success_matches"]
                for report in scenario_reports
            ),
            "all_metric_rows_are_registered": all(
                report["metric_governance"]["metric_rows_are_registered"]
                for report in scenario_reports
            ),
            "full_graph_is_baseline_not_oracle_all": all(
                report["checks"]["full_graph_is_baseline_not_oracle"]
                for report in scenario_reports
            ),
            "training_run": False,
            "v5_code_migrated": False,
        },
    }


def _row_from_evaluation(
    name: str,
    family: str,
    decision_source: str,
    evaluation: TopologyEvaluation,
    is_oracle: bool,
    decision_diagnostics: Mapping[str, object],
) -> BaselineReportRow:
    require_registered_metrics(evaluation.metrics.keys())
    return BaselineReportRow(
        name=name,
        family=family,
        decision_source=decision_source,
        topology_id=evaluation.topology_id,
        selected_edge_ids=evaluation.selected_edge_ids,
        is_oracle=is_oracle,
        metrics=dict(evaluation.metrics),
        metric_rows=evaluation.metric_rows(),
        decision_diagnostics=decision_diagnostics,
    )


def _oracle_payload(oracle_result) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": oracle_result.status,
        "oracle_name": oracle_result.oracle_name,
        "selected_edge_ids": list(oracle_result.selected_edge_ids),
        "searched_topology_count": oracle_result.searched_topology_count,
        "is_exhaustive": oracle_result.is_exhaustive,
        "is_deployment_actor_input": False,
    }
    if oracle_result.evaluation is not None:
        require_registered_metrics(oracle_result.evaluation.metrics.keys())
        payload["topology_id"] = oracle_result.evaluation.topology_id
        payload["metrics"] = dict(oracle_result.evaluation.metrics)
        payload["metric_rows"] = [
            dict(row) for row in oracle_result.evaluation.metric_rows()
        ]
    else:
        payload["topology_id"] = None
        payload["metrics"] = None
        payload["metric_rows"] = []
    return payload


def _format_probability_tag(value: float) -> str:
    return f"p{int(round(value * 100)):02d}"


def _full_graph_is_not_oracle(
    baseline_rows: list[BaselineReportRow],
    oracle_payload: Mapping[str, object],
) -> bool:
    full_rows = [row for row in baseline_rows if row.name == "full"]
    if not full_rows:
        return False
    return (
        full_rows[0].is_oracle is False
        and full_rows[0].topology_id == "baseline:full"
        and oracle_payload.get("oracle_name") != "full"
    )


def _full_graph_success_matches(
    baseline_rows: list[BaselineReportRow],
    expected: bool,
) -> bool:
    full_rows = [row for row in baseline_rows if row.name == "full"]
    if not full_rows:
        return False
    return bool(full_rows[0].metrics["consensus_success"]) is expected
