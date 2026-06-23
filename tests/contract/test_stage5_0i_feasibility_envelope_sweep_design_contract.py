import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_doc(name: str) -> str:
    return _read(ROOT / "docs" / name)


def test_stage5_0i_replay_script_prints_design_manifest_without_sweep_execution() -> None:
    script = ROOT / "scripts" / "replay" / "stage5_0i_feasibility_envelope_sweep_design.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert '"stage": "stage_5_0i_feasibility_envelope_sweep_design"' in completed.stdout
    assert '"tau_requirement_min": 0.9' in completed.stdout
    assert '"required_sweeps_present": true' in completed.stdout
    assert '"final_tau_selected": false' in completed.stdout
    assert '"simulation_parameters_changed_to_force_feasibility": false' in completed.stdout
    assert '"reward_implemented": false' in completed.stdout
    assert '"training_run": false' in completed.stdout
    assert '"v5_code_migrated": false' in completed.stdout


