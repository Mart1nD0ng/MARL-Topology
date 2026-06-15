"""Print the Stage 4.5 baseline and oracle-candidate review payload."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_topology.evaluation import build_stage4_5_baseline_oracle_review


def main() -> None:
    print(json.dumps(build_stage4_5_baseline_oracle_review(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
