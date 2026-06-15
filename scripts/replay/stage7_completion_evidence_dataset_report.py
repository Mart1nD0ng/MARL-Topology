#!/usr/bin/env python3
"""Build the Stage 7 completion evidence dataset and optional artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from marl_topology.data import (  # noqa: E402
    build_stage7_completion_report,
    write_stage7_completion_artifact,
)


def build_stage7_completion_payload(
    *,
    write_artifact: bool,
    include_rows: bool,
) -> dict[str, object]:
    build = build_stage7_completion_report()
    payload = dict(build.report) if include_rows else _compact_payload(build.report)
    payload["artifact_written"] = False
    payload["owner_decision_required_for_stage8"] = True
    if write_artifact:
        payload["artifact_write_result"] = write_stage7_completion_artifact(
            project_root=ROOT,
            owner_approved_evidence_export=True,
        )
        payload["artifact_written"] = True
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-artifact",
        action="store_true",
        help="write manifest-validated evidence-only artifact under result_save",
    )
    parser.add_argument(
        "--include-rows",
        action="store_true",
        help="include full evidence rows in stdout; artifacts always contain the dataset",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_stage7_completion_payload(
                write_artifact=args.write_artifact,
                include_rows=args.include_rows,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _compact_payload(report: dict[str, object]) -> dict[str, object]:
    payload = dict(report)
    rows = payload.pop("scenario_topology_evidence_rows", [])
    payload["scenario_topology_evidence_summary"] = {
        "row_count": len(rows) if isinstance(rows, list) else payload.get("row_count", 0),
        "full_rows_omitted_from_stdout": True,
        "use_include_rows_for_verbose_stdout": True,
    }
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
