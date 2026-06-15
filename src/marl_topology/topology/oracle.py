"""Small-graph topology oracle and baseline evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from marl_topology.policies import PolicyBaselines
from marl_topology.topology.evaluator import TopologyEvaluation, TopologyEvaluator


@dataclass(frozen=True, slots=True)
class OracleResult:
    status: str
    oracle_name: str | None
    selected_edge_ids: tuple[str, ...]
    evaluation: TopologyEvaluation | None
    baseline_evaluations: dict[str, TopologyEvaluation]
    searched_topology_count: int
    is_exhaustive: bool


class TopologyOracle:
    """Exhaustive small-graph oracle.

    Full graph remains a baseline. The oracle label is reserved for the
    exhaustive search result and must not be used as a deployment actor input.
    """

    def __init__(self, evaluator: TopologyEvaluator, max_exhaustive_edges: int = 10) -> None:
        if max_exhaustive_edges < 0:
            raise ValueError("max_exhaustive_edges must be nonnegative")
        self.evaluator = evaluator
        self.max_exhaustive_edges = max_exhaustive_edges

    def evaluate_baselines(self, random_seed: int = 0) -> dict[str, TopologyEvaluation]:
        graph = self.evaluator.graph
        link_records = self.evaluator.link_records
        baselines = (
            PolicyBaselines.empty(graph),
            PolicyBaselines.full(graph),
            PolicyBaselines.greedy_reliability(graph, link_records),
            PolicyBaselines.random(graph, seed=random_seed),
        )
        return {
            baseline.name: self.evaluator.evaluate(
                set(baseline.edge_ids),
                topology_id=f"baseline:{baseline.name}",
            )
            for baseline in baselines
        }

    def solve(self, random_seed: int = 0) -> OracleResult:
        baseline_evaluations = self.evaluate_baselines(random_seed=random_seed)
        edge_ids = self.evaluator.graph.edge_ids
        if len(edge_ids) > self.max_exhaustive_edges:
            return OracleResult(
                status="unresolved",
                oracle_name=None,
                selected_edge_ids=(),
                evaluation=None,
                baseline_evaluations=baseline_evaluations,
                searched_topology_count=0,
                is_exhaustive=False,
            )

        searched = 0
        feasible: list[TopologyEvaluation] = []
        for size in range(len(edge_ids) + 1):
            for selected in combinations(edge_ids, size):
                searched += 1
                evaluation = self.evaluator.evaluate(
                    set(selected),
                    topology_id=f"oracle_candidate:{searched}",
                )
                if evaluation.consensus.consensus_success:
                    feasible.append(evaluation)

        if not feasible:
            return OracleResult(
                status="infeasible",
                oracle_name="oracle_exhaustive",
                selected_edge_ids=(),
                evaluation=None,
                baseline_evaluations=baseline_evaluations,
                searched_topology_count=searched,
                is_exhaustive=True,
            )

        best = min(
            feasible,
            key=lambda item: (
                float(item.metrics["latency"]),
                float(item.metrics["energy"]),
                int(item.diagnostics["edge_count"]),
            ),
        )
        return OracleResult(
            status="feasible",
            oracle_name="oracle_exhaustive",
            selected_edge_ids=best.selected_edge_ids,
            evaluation=best,
            baseline_evaluations=baseline_evaluations,
            searched_topology_count=searched,
            is_exhaustive=True,
        )
