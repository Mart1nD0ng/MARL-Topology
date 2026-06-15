"""Stage 31 Phase F: large-scale production-readiness test entry point.

Builds the production scenario dataset, trains the integrated stack (constraint-
aware scorer -> budget-aware sampler -> assembler projection -> expected-initiator
PBFT evaluator -> feasibility-first surrogate -> policy gradient), scores the
Stage 26-30 blockers, and writes diagnostic artifacts under logs/ (gitignored).

Usage:
    python scripts/train/stage31_production_readiness_test.py [scenario_count]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_topology.data.stage31_production_dataset import (  # noqa: E402
    build_dataset_manifest,
    build_production_dataset,
)
from marl_topology.data.stage31_scenario_generator import (  # noqa: E402
    ProductionScenarioConfig,
)
from marl_topology.training.stage31_readiness import (  # noqa: E402
    Stage31ReadinessConfig,
    build_readiness_scorecard,
    train_readiness,
)

# Diagnostic artifacts go under logs/ (gitignored), not result_save/, which is
# governed by frozen allowlists. This mirrors the Stage 29/30 diagnostic pattern
# where readiness evidence lives outside the evidence/training artifact scope.
ARTIFACT_DIR = ROOT / "logs" / "stage31_production_readiness" / "stage31_readiness_v1"


def run(scenario_count: int = 150) -> dict[str, object]:
    dataset = build_production_dataset(
        ProductionScenarioConfig(seed=31, scenario_count=scenario_count)
    )
    result = train_readiness(dataset, Stage31ReadinessConfig(seed=31))
    scorecard = build_readiness_scorecard(result, dataset)
    payload = {
        "manifest": build_dataset_manifest(dataset),
        "dataset_quality": dict(dataset.quality_report),
        "training": {
            "surrogate_config_id": result.surrogate_config_id,
            "before_eval": result.before_eval,
            "warm_start_eval": result.warm_start_eval,
            "after_eval": result.after_eval,
            "test_eval": result.test_eval,
            "teacher_feasible_rate": result.teacher_feasible_rate,
            "history": result.history,
        },
        "scorecard": scorecard,
    }
    return payload


def write_artifacts(payload: dict[str, object]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "manifest.json").write_text(
        json.dumps(payload["manifest"], indent=2), encoding="utf-8"
    )
    (ARTIFACT_DIR / "dataset_quality.json").write_text(
        json.dumps(payload["dataset_quality"], indent=2, default=str), encoding="utf-8"
    )
    (ARTIFACT_DIR / "training_report.json").write_text(
        json.dumps(payload["training"], indent=2, default=str), encoding="utf-8"
    )
    (ARTIFACT_DIR / "readiness_scorecard.json").write_text(
        json.dumps(payload["scorecard"], indent=2, default=str), encoding="utf-8"
    )


def main() -> None:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    payload = run(count)
    write_artifacts(payload)
    scorecard = payload["scorecard"]
    print(f"verdict: {scorecard['verdict']}")
    print(f"all_blockers_resolved: {scorecard['all_stage26_30_blockers_resolved']}")
    for name, block in scorecard["components"].items():
        print(f"  [{block['status']:8s}] {name}: {block['evidence']}")
    print(f"artifacts: {ARTIFACT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
