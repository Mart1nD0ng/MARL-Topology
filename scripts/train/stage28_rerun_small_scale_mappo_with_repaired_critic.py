#!/usr/bin/env python3
"""Run Stage 28 small-scale MAPPO rerun with the repaired Stage 27 critic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_topology.training.mappo.stage28_repaired_critic_pilot import (  # noqa: E402
    Stage28PilotConfig,
    run_stage28_repaired_critic_small_scale_pilot,
    write_stage28_training_artifacts,
)


def build_report(*, write_artifacts: bool = True) -> dict[str, object]:
    report = run_stage28_repaired_critic_small_scale_pilot(
        config=Stage28PilotConfig(),
        project_root=ROOT,
    )
    if write_artifacts:
        artifact_report = write_stage28_training_artifacts(report, project_root=ROOT)
        report = {**report, "artifact_written": True, "artifact_report": artifact_report}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-write-artifacts",
        action="store_true",
        help="Run the fixed Stage 28 pilot without writing report artifacts.",
    )
    args = parser.parse_args()
    report = build_report(write_artifacts=not args.no_write_artifacts)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if bool(report.get("pass_gate")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
