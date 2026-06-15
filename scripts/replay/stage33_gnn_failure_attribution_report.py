"""Replay the Stage 33 report into a failure-attribution summary.

This script does not train. It reads the manifest-validated Stage33 report and
classifies the pass/fail gate into owner-decision categories.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_topology.training.production_mappo_adapter import (  # noqa: E402
    STAGE33_ARTIFACT_ROOT,
    failure_review,
)


REPORT_PATH = ROOT / STAGE33_ARTIFACT_ROOT / "training_report.json"
OUTPUT_PATH = ROOT / STAGE33_ARTIFACT_ROOT / "failure_attribution_report.json"


def build_report() -> dict[str, object]:
    if not REPORT_PATH.exists():
        raise SystemExit(
            "Stage33 training report is missing. Run "
            "python scripts/train/stage33_gnn_stability_repair_training.py first."
        )
    source = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    review = failure_review(
        source["pass_fail_gate"],
        source["selection"],
    )
    payload = {
        "stage": source["stage"],
        "verdict": source["verdict"],
        "pass_gate": source["pass_gate"],
        "failure_review": review,
        "mlp_promoted": False,
        "owner_decision_required": True,
    }
    OUTPUT_PATH.write_bytes(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
    return payload


def main() -> None:
    print(json.dumps(build_report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
