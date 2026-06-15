#!/usr/bin/env python3
"""Print the Stage 2.5 scenario fixture report without writing files."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.dont_write_bytecode = True
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from marl_topology.evaluation import build_stage2_scenario_fixture_report  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(build_stage2_scenario_fixture_report(), indent=2, sort_keys=True))
