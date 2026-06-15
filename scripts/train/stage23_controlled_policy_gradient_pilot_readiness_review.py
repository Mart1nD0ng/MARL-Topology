#!/usr/bin/env python3
"""Run Stage 23 policy-gradient readiness review without training."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_topology.evaluation.stage23_policy_gradient_readiness import (  # noqa: E402
    run_stage23_controlled_policy_gradient_pilot_readiness_review,
)


def main() -> int:
    report = run_stage23_controlled_policy_gradient_pilot_readiness_review(
        project_root=ROOT
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["cleanup_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
