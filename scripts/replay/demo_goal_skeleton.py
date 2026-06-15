#!/usr/bin/env python3
"""Run the Stage 2 goal-skeleton demo without writing result files."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.dont_write_bytecode = True
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from marl_topology.evaluation import build_demo_report  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(build_demo_report(), indent=2, sort_keys=True))
