#!/usr/bin/env python3
"""Stage 21 objective-stack-aligned assembler-aware rerun."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_topology.evaluation.stage21_fair_evaluation import (  # noqa: E402
    Stage21FairEvaluationConfig,
    run_stage21_objective_stack_aligned_assembler_aware_rerun,
)


def build_report() -> dict[str, object]:
    """Run Stage 21 and return a JSON-serializable report."""

    return run_stage21_objective_stack_aligned_assembler_aware_rerun(
        config=Stage21FairEvaluationConfig()
    )


def main() -> int:
    print(json.dumps(build_report(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
