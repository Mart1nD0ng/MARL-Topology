#!/usr/bin/env python3
"""Stage 20 supervised actor policy evaluation through the assembler."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_topology.evaluation.stage20_supervised_actor_policy_evaluation import (  # noqa: E402
    Stage20PolicyEvaluationConfig,
    run_stage20_supervised_actor_policy_evaluation,
)


def build_report() -> dict[str, object]:
    """Run Stage 20 evaluation and return a JSON-serializable report."""

    return run_stage20_supervised_actor_policy_evaluation(
        config=Stage20PolicyEvaluationConfig()
    )


def main() -> int:
    print(json.dumps(build_report(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
