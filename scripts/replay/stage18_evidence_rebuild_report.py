"""Print the Stage 18 evidence rebuild report as JSON.

This is a report-only replay script. It does not write artifacts or execute
training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from marl_topology.data.learning_evidence_stage18 import (  # noqa: E402
    build_stage18_evidence_rebuild_report,
)


def main() -> int:
    build = build_stage18_evidence_rebuild_report()
    print(json.dumps(build.report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
