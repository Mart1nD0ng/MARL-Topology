#!/usr/bin/env python3
"""Print the Stage 5.3 normalization-reference selection report."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from marl_topology.evaluation import build_stage5_3_normalization_reference_selection


def main() -> int:
    print(json.dumps(build_stage5_3_normalization_reference_selection(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
