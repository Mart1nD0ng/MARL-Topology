#!/usr/bin/env python3
"""Stage 19 supervised actor stack rerun on Stage 18 evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_topology.training.stage19_supervised_actor_stack import (  # noqa: E402
    Stage19SupervisedActorStackConfig,
    run_stage19_supervised_actor_stack,
)


def build_report() -> dict[str, object]:
    """Run the bounded supervised actor-only Stage 19 stack and return JSON data."""

    return run_stage19_supervised_actor_stack(
        config=Stage19SupervisedActorStackConfig()
    )


def main() -> int:
    print(json.dumps(build_report(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
