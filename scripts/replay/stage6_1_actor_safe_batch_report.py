#!/usr/bin/env python3
"""Print the Stage 6.1 actor-safe batch report without model or training."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from marl_topology.data import build_stage6_1_actor_safe_batch_report  # noqa: E402
from marl_topology.env import ActorObservation  # noqa: E402
from marl_topology.training import build_stage6_0_minimal_stack_report  # noqa: E402


def main() -> int:
    stack_report = build_stage6_0_minimal_stack_report(project_root=ROOT)
    observations = (
        ActorObservation(
            agent_id="veh_0",
            agent_kind="vehicle",
            time_step=0,
            local_position_m=(0.0, 0.0, 1.5),
        ),
        ActorObservation(
            agent_id="rsu_0",
            agent_kind="rsu",
            time_step=0,
            local_position_m=(10.0, 0.0, 6.0),
        ),
    )
    report = build_stage6_1_actor_safe_batch_report(
        observations,
        stack_ready=bool(stack_report["stack_ready"]),
    )
    report["stage6_0_guard"] = {
        "stack_ready": stack_report["stack_ready"],
        "writes_performed": stack_report["writes_performed"],
        "training_execution_allowed": stack_report["training_execution_allowed"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
