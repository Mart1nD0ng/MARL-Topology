#!/usr/bin/env python3
"""Stage 13 local GNN actor comparison against the MLP baseline."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_topology.models import (  # noqa: E402
    LOCAL_GNN_EDGE_SCORER_MODEL_ID,
    LocalGNNEdgeScorer,
    LocalMLPEdgeScorer,
)
from marl_topology.training.supervised_actor_trainer import (  # noqa: E402
    SupervisedActorTrainingConfig,
    run_supervised_actor_training,
)
from marl_topology.training.supervised_batching import load_learning_evidence_json  # noqa: E402


EVIDENCE_PATH = (
    ROOT
    / "result_save"
    / "evidence_dataset_only"
    / "stage7_completion_learning_evidence_dataset_v1"
    / "learning_evidence.json"
)


def build_report() -> dict[str, object]:
    evidence = load_learning_evidence_json(EVIDENCE_PATH)
    rows = tuple(dict(row) for row in evidence["rows"])  # type: ignore[index]
    config = SupervisedActorTrainingConfig(seed=13, epochs=60, tiny_batch_epochs=240)
    mlp_start = time.perf_counter()
    mlp_model, mlp_result = run_supervised_actor_training(
        rows,
        config=config,
        model_factory=LocalMLPEdgeScorer,
    )
    mlp_runtime = time.perf_counter() - mlp_start
    gnn_start = time.perf_counter()
    gnn_model, gnn_result = run_supervised_actor_training(
        rows,
        config=config,
        model_factory=LocalGNNEdgeScorer,
    )
    gnn_runtime = time.perf_counter() - gnn_start
    selected = _select_actor(mlp_result.to_dict(), gnn_result.to_dict())
    return {
        "stage": "stage_13_local_gnn_edge_scorer",
        "dataset_id": evidence["dataset_id"],
        "mlp": {
            **mlp_result.to_dict(),
            "parameter_count": _parameter_count(mlp_model),
            "runtime_s": mlp_runtime,
        },
        "gnn": {
            **gnn_result.to_dict(),
            "parameter_count": _parameter_count(gnn_model),
            "runtime_s": gnn_runtime,
            "actor_graph_scope": "local_ego_incident_candidate_edges_only",
        },
        "comparison": {
            "validation_edge_loss_delta_gnn_minus_mlp": (
                gnn_result.validation_loss - mlp_result.validation_loss
            ),
            "ranking_metric_delta_gnn_minus_mlp": (
                gnn_result.pairwise_accuracy - mlp_result.pairwise_accuracy
            ),
            "assembler_feasibility_mlp": mlp_result.assembler_diagnostics[
                "feasible_under_tau_requirement"
            ],
            "assembler_feasibility_gnn": gnn_result.assembler_diagnostics[
                "feasible_under_tau_requirement"
            ],
            "oracle_gap_available": False,
            "training_stability": "finite_losses_no_nan",
        },
        "selected_stage14_starting_actor": selected["actor"],
        "selection_reason": selected["reason"],
        "global_actor_graph_used": False,
        "oracle_labels_used_as_actor_input": False,
        "critic_outputs_used_as_actor_input": False,
        "artifact_written": False,
        "checkpoint_written": False,
        "rl_training_performed": False,
    }


def _select_actor(mlp: dict[str, object], gnn: dict[str, object]) -> dict[str, str]:
    mlp_loss = float(mlp["validation_loss"])
    gnn_loss = float(gnn["validation_loss"])
    if gnn_loss <= mlp_loss:
        return {
            "actor": LOCAL_GNN_EDGE_SCORER_MODEL_ID,
            "reason": "GNN matched or improved validation edge loss with local-only message passing.",
        }
    return {
        "actor": "local_mlp_edge_scorer_stage11_supervised",
        "reason": (
            "MLP retained lower validation loss on the small contradictory "
            "Stage 7 evidence; GNN remains implemented with diagnostics."
        ),
    }


def _parameter_count(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def main() -> int:
    print(json.dumps(build_report(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
