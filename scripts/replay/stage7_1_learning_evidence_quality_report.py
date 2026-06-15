#!/usr/bin/env python3
"""Print a Stage 7.1 learning evidence data quality report."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from marl_topology.data import (  # noqa: E402
    build_learning_evidence_dataset,
    evaluate_learning_evidence_quality,
)
from marl_topology.evaluation.demo import build_demo_stack  # noqa: E402
from marl_topology.policies import build_local_observations  # noqa: E402


def build_stage7_1_demo_quality_report() -> dict[str, object]:
    scene, graph, evaluator, oracle = build_demo_stack()
    observations = build_local_observations(
        scene=scene,
        graph=graph,
        link_records=evaluator.link_records,
        time_step=0,
    )
    baselines = oracle.evaluate_baselines(random_seed=7)
    variants: dict[str, tuple[str, ...]] = {
        "weak_disconnected_baseline": baselines["empty"].selected_edge_ids,
        "sparse_candidate": baselines["greedy_reliability"].selected_edge_ids,
        "dense_full_graph_baseline": baselines["full"].selected_edge_ids,
    }
    oracle_result = oracle.solve(random_seed=7)
    if oracle_result.evaluation is not None:
        variants["oracle_candidate_diagnostic"] = oracle_result.evaluation.selected_edge_ids

    dataset = build_learning_evidence_dataset(
        evaluator,
        observations=observations,
        topology_variants=variants,
        tau_requirement_min=0.9,
    )
    report = evaluate_learning_evidence_quality(dataset)
    payload = report.to_dict()
    payload["quality_scope"] = "minimal_stage7_evidence_sensor"
    payload["dataset_is_training_sufficient"] = False
    payload["stage7_closed"] = report.stage7_exit_ready
    payload["owner_decision_required"] = True
    return payload


def main() -> int:
    print(json.dumps(build_stage7_1_demo_quality_report(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
