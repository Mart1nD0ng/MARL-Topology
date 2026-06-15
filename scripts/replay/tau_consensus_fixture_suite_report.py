from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from marl_topology.evaluation import build_stage5_0f_tau_calibration_fixture_suite_report


def main() -> None:
    report = build_stage5_0f_tau_calibration_fixture_suite_report()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
