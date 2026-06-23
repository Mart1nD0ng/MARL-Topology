import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_doc(name: str) -> str:
    return _read(ROOT / "docs" / name)


def test_stage5_0f_replay_script_prints_fixture_suite_report() -> None:
    script = ROOT / "scripts" / "replay" / "tau_consensus_fixture_suite_report.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert '"stage": "stage_5_0f_minimal_tau_calibration_fixture_suite"' in completed.stdout
    assert '"required_families_present": true' in completed.stdout
    assert '"training_run": false' in completed.stdout
    assert '"v5_code_migrated": false' in completed.stdout


def test_stage5_0d_replay_script_can_use_stage5_0f_fixture_source() -> None:
    script = ROOT / "scripts" / "replay" / "tau_consensus_calibration_report.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--source", "stage5_0f", "--tau", "0.05", "--tau", "0.5"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert '"source_kind": "stage5_0f_alpha_fixture_suite"' in completed.stdout
    assert '"source_is_alpha_fixture_suite": true' in completed.stdout
    assert '"tau_selected": false' in completed.stdout
    assert '"final_tau_consensus": null' in completed.stdout


