import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_doc(name: str) -> str:
    return _read(ROOT / "docs" / name)


def test_stage5_0k_replay_script_prints_stage3_backed_report() -> None:
    script = ROOT / "scripts" / "replay" / "stage5_0k_stage3_backed_feasibility_envelope_sweep.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert '"stage": "stage_5_0k_stage3_backed_feasibility_envelope_sweep_hardening"' in completed.stdout
    assert '"tau_requirement_min": 0.9' in completed.stdout
    assert '"bandwidth_sweep"' in completed.stdout
    assert '"deadline_sweep"' in completed.stdout
    assert '"resource_orthogonalization_sweep"' in completed.stdout
    assert '"all_rows_stage3_backed": true' in completed.stdout
    assert '"all_rows_use_finite_blocklength": true' in completed.stdout
    assert '"all_rows_use_stage3_network_records": true' in completed.stdout
    assert '"resource_orthogonalization_improves_reliability": true' in completed.stdout
    assert '"final_tau_selected": false' in completed.stdout
    assert '"reward_implemented": false' in completed.stdout
    assert '"training_run": false' in completed.stdout
    assert '"v5_code_migrated": false' in completed.stdout


def test_stage5_0k_source_avoids_sampling_reward_training_model_and_v5_code() -> None:
    source = _read(
        ROOT
        / "src"
        / "marl_topology"
        / "evaluation"
        / "feasibility_envelope_sweep_stage3_backed.py"
    )
    banned_terms = [
        "random",
        "sample",
        "Monte Carlo",
        "combinations(",
        "class Actor(",
        "class Critic(",
        "COMA",
        "MAPPO",
        "optimizer",
        "backward(",
        "train_loop",
        "def reward",
        "def compute_reward",
        "class Reward",
        "reward =",
        "reward:",
        "P_eff_soft",
        "P_eff_hard",
        "hard_eval",
        "soft_train",
        "D:\\PhD_works\\v5",
    ]
    hits = [term for term in banned_terms if term in source]
    assert not hits, f"Stage 5.0k source added forbidden terms: {hits}"
