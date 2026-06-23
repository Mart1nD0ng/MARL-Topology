import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_doc(name: str) -> str:
    return _read(ROOT / "docs" / name)


def test_stage5_0l_replay_script_prints_range_review_report() -> None:
    script = ROOT / "scripts" / "replay" / "stage5_0l_stage3_backed_sweep_range_review.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert '"stage": "stage_5_0l_stage3_backed_sweep_range_expansion_and_realism_review"' in completed.stdout
    assert '"tau_requirement_min": 0.9' in completed.stdout
    assert '"tx_power_sweep"' in completed.stdout
    assert '"payload_sweep"' in completed.stdout
    assert '"rsu_height_placement_sweep"' in completed.stdout
    assert '"resource_budget_limit_sweep"' in completed.stdout
    assert '"every_sweep_has_infeasible_to_feasible_transition": true' in completed.stdout
    assert '"all_rows_stage3_backed": true' in completed.stdout
    assert '"unknown_needs_reference"' in completed.stdout
    assert '"final_tau_selected": false' in completed.stdout
    assert '"reward_implemented": false' in completed.stdout
    assert '"training_run": false' in completed.stdout
    assert '"v5_code_migrated": false' in completed.stdout


def test_stage5_0l_source_avoids_sampling_reward_training_model_and_v5_code() -> None:
    source = _read(
        ROOT
        / "src"
        / "marl_topology"
        / "evaluation"
        / "feasibility_envelope_sweep_range_review.py"
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
    assert not hits, f"Stage 5.0l source added forbidden terms: {hits}"
