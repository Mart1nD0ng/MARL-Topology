#!/usr/bin/env python3
"""Print the Stage 17 actor-label disambiguation report.

This script is report-only. It does not train, create checkpoints, or write
artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from marl_topology.data import build_stage17_actor_label_disambiguation_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Stage 17 actor-label disambiguation report."
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=8,
        help="Maximum contradiction clusters to print; use a negative value for all.",
    )
    args = parser.parse_args()

    report = build_stage17_actor_label_disambiguation_report().to_dict()
    if args.max_clusters >= 0:
        report["contradiction_clusters"] = report["contradiction_clusters"][
            : args.max_clusters
        ]
        report["cluster_output_truncated"] = True
        report["cluster_output_limit"] = args.max_clusters
    else:
        report["cluster_output_truncated"] = False
        report["cluster_output_limit"] = "all"
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
