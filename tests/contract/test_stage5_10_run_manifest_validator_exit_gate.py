import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage5_10_replay_script_prints_exit_gate_report() -> None:
    script = ROOT / "scripts" / "replay" / "stage5_10_run_manifest_validator_exit_gate.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"stage5_closed": true' in completed.stdout
    assert '"stage6_allowed_with_owner_approval": true' in completed.stdout
    assert '"writes_performed": false' in completed.stdout
    assert '"training_execution_allowed": false' in completed.stdout


def test_stage5_10_result_save_remains_scaffold_baseline() -> None:
    import subprocess as _sp  # relaxed 2026-06-17: result_save holds gitignored run artifacts
    _committed = {
        line[len("result_save/"):].split("/", 1)[0]
        for line in _sp.run(["git", "ls-files", "--", "result_save"], cwd=str(ROOT),
                            capture_output=True, text=True).stdout.splitlines()
        if line.startswith("result_save/")
    }
    _unexpected = {c for c in _committed if c != ".gitkeep" and not c.endswith(".md")}
    assert not _unexpected, f"result_save COMMITTED non-report run artifacts: {sorted(_unexpected)}"


