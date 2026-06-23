import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_doc(name: str) -> str:
    return _read(ROOT / "docs" / name)


def test_stage5_0d_source_and_replay_script_exist_with_explicit_tau_only() -> None:
    source = ROOT / "src" / "marl_topology" / "evaluation" / "tau_consensus_calibration_report.py"
    script = ROOT / "scripts" / "replay" / "tau_consensus_calibration_report.py"
    assert source.exists()
    assert script.exists()

    source_text = _read(source)
    script_text = _read(script)
    assert "build_tau_consensus_calibration_report" in source_text
    assert "required=True" in script_text
    assert "--tau" in script_text
    assert "reliability_threshold = 0.2" not in source_text
    assert "tau_candidates: Iterable[float | TauCandidate]" in source_text


def test_stage5_0d_replay_script_requires_tau_and_prints_json() -> None:
    script = ROOT / "scripts" / "replay" / "tau_consensus_calibration_report.py"
    missing_tau = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing_tau.returncode != 0
    assert "--tau" in missing_tau.stderr

    with_tau = subprocess.run(
        [sys.executable, str(script), "--tau", "0.15", "--tau", "0.55"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert '"final_tau_consensus": null' in with_tau.stdout
    assert '"tau_selected": false' in with_tau.stdout
    assert '"source_scope": "stage4_8_smoke_test_only_not_calibration_set"' in with_tau.stdout


