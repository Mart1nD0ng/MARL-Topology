#!/usr/bin/env python3
"""Stage 10 supervised loss dry-run report with no parameter update."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_topology.training.supervised_loss_dry_run_report import (  # noqa: E402
    build_stage10_supervised_loss_dry_run_report,
)


EVIDENCE_PATH = (
    ROOT
    / "result_save"
    / "evidence_dataset_only"
    / "stage7_completion_learning_evidence_dataset_v1"
    / "learning_evidence.json"
)


def build_report() -> dict[str, object]:
    return build_stage10_supervised_loss_dry_run_report(EVIDENCE_PATH)


def main() -> int:
    print(json.dumps(build_report(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
