from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from marl_topology.evaluation import build_stage5_0m_objective_readiness_review  # noqa: E402


def main() -> None:
    report = build_stage5_0m_objective_readiness_review()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
