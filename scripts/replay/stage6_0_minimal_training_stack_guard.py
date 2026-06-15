#!/usr/bin/env python3
"""Print the Stage 6.0 minimal training-stack guard report."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from marl_topology.training import build_stage6_0_minimal_stack_report  # noqa: E402


def main() -> int:
    report = build_stage6_0_minimal_stack_report(project_root=ROOT)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
