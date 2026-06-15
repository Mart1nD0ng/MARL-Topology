#!/usr/bin/env python3
"""Stage 14 temporal actor ablation: GRU sanity before LSTM replacement."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_topology.training.temporal_actor_ablation import (  # noqa: E402
    build_stage14_temporal_actor_ablation_report,
)


EVIDENCE_PATH = (
    ROOT
    / "result_save"
    / "evidence_dataset_only"
    / "stage7_completion_learning_evidence_dataset_v1"
    / "learning_evidence.json"
)


def build_report() -> dict[str, object]:
    return build_stage14_temporal_actor_ablation_report(EVIDENCE_PATH)


def main() -> int:
    print(json.dumps(build_report(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
