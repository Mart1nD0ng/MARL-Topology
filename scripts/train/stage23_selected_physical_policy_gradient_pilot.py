#!/usr/bin/env python3
"""Stage 23 selected-physical policy-gradient sampler trial."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_topology.training.policy_gradient.pilot_runner import (  # noqa: E402
    STAGE23_PASS_VERDICT,
    Stage23PolicyGradientConfig,
    build_stage23_manifest,
    run_stage23_selected_physical_policy_gradient_pilot,
)
from marl_topology.training.run_manifest_validator import (  # noqa: E402
    validate_run_manifest_dry_run,
)


def build_report() -> dict[str, object]:
    manifest = build_stage23_manifest("stage23_selected_physical_pg_micro_pilot")
    validation = validate_run_manifest_dry_run(manifest, project_root=ROOT)
    if not validation.is_valid:
        raise RuntimeError(f"manifest validation failed: {validation.error_codes()}")
    report = run_stage23_selected_physical_policy_gradient_pilot(
        config=Stage23PolicyGradientConfig(),
        project_root=ROOT,
    )
    report["manifest_validation"] = validation.to_dict()
    report["manifest"] = manifest
    return report


def main() -> int:
    report = build_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["verdict"] == STAGE23_PASS_VERDICT else 1


if __name__ == "__main__":
    raise SystemExit(main())
