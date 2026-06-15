#!/usr/bin/env python3
"""Stage 24 critic-integrated MAPPO-style micro-loop runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_topology.training.mappo.trainer import (  # noqa: E402
    STAGE24_FAIL_VERDICT,
    Stage24LoopConfig,
    build_stage24_manifest,
    run_stage24_critic_integrated_micro_loop,
)
from marl_topology.training.run_manifest_validator import (  # noqa: E402
    validate_run_manifest_dry_run,
)


def build_report(mode: str) -> dict[str, object]:
    config = Stage24LoopConfig.for_mode(mode)
    manifest = build_stage24_manifest(f"stage24_{mode}_critic_integrated_micro_loop")
    validation = validate_run_manifest_dry_run(manifest, project_root=ROOT)
    if not validation.is_valid:
        raise RuntimeError(f"manifest validation failed: {validation.error_codes()}")
    report = run_stage24_critic_integrated_micro_loop(
        mode=mode,
        config=config,
        project_root=ROOT,
    )
    report["manifest_validation"] = validation.to_dict()
    report["manifest"] = manifest
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "micro"), default="smoke")
    args = parser.parse_args()
    report = build_report(args.mode)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["verdict"] != STAGE24_FAIL_VERDICT else 1


if __name__ == "__main__":
    raise SystemExit(main())
