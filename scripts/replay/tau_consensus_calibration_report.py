from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from marl_topology.evaluation import (
    build_stage5_0f_tau_calibration_fixture_suite_report,
    build_tau_consensus_calibration_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print the Stage 5.0d tau-consensus calibration report without "
            "selecting final tau_consensus."
        )
    )
    parser.add_argument(
        "--tau",
        action="append",
        type=float,
        required=True,
        help="Owner-declared candidate tau value. Repeat for multiple candidates.",
    )
    parser.add_argument(
        "--tau-source",
        default="owner_declared_cli",
        help="Source label for the supplied tau candidates.",
    )
    parser.add_argument(
        "--tau-owner-note",
        default="",
        help="Optional owner note copied into candidate tau rows.",
    )
    parser.add_argument(
        "--source",
        choices=("stage4_8", "stage5_0f"),
        default="stage4_8",
        help="Report source rows to evaluate.",
    )
    args = parser.parse_args()
    source_report = None
    if args.source == "stage5_0f":
        source_report = build_stage5_0f_tau_calibration_fixture_suite_report()
    report = build_tau_consensus_calibration_report(
        args.tau,
        tau_source=args.tau_source,
        tau_owner_note=args.tau_owner_note,
        source_report=source_report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
