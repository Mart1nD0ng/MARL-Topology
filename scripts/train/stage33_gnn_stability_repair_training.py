"""Run the Stage 33 fixed GNN stability repair protocol.

The active production path is the Stage33 adapter over the official Stage24/25/28
mappo components. This script intentionally runs a bounded protocol: three fixed
GNN stability configs, five seeds, the active/diagnostic GNN variants, and an MLP
diagnostic baseline on the same graph-structure dataset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_topology.data.stage33_graph_structure_dataset import (  # noqa: E402
    Stage33GraphStructureConfig,
    build_stage33_graph_structure_dataset,
)
from marl_topology.training.production_mappo_adapter import (  # noqa: E402
    Stage33ProductionMappoAdapter,
    stage33_fixed_gnn_stability_configs,
    write_stage33_training_artifacts,
)


def run() -> dict[str, object]:
    dataset = build_stage33_graph_structure_dataset(
        Stage33GraphStructureConfig(
            seed=33,
            scenario_count=7,
            node_count_choices=(6, 7, 8, 9, 10),
        )
    )
    configs = stage33_fixed_gnn_stability_configs(
        seeds=(3301, 3302, 3303, 3304, 3305)
    )
    adapter = Stage33ProductionMappoAdapter()
    report = adapter.run_fixed_protocol(
        dataset=dataset,
        stability_configs=configs,
        project_root=ROOT,
        critic_epochs=1,
    )
    artifact = write_stage33_training_artifacts(report, project_root=ROOT)
    return {**report, "artifact_write": artifact}


def main() -> None:
    report = run()
    gate = report["pass_fail_gate"]  # type: ignore[index]
    selection = report["selection"]  # type: ignore[index]
    artifact = report["artifact_write"]  # type: ignore[index]
    print(json.dumps(
        {
            "stage": report["stage"],
            "verdict": report["verdict"],
            "pass_gate": report["pass_gate"],
            "selected_model_id": selection.get("selected_model_id"),
            "selected_config_id": selection.get("selected_config_id"),
            "gate_issues": gate.get("issues"),
            "artifact_dir": artifact.get("artifact_dir"),
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
