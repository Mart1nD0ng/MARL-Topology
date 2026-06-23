import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_doc(name: str) -> str:
    return _read(ROOT / "docs" / name)


def test_stage5_0m_replay_script_prints_readiness_report() -> None:
    script = ROOT / "scripts" / "replay" / "stage5_0m_objective_readiness_review.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert '"stage": "stage_5_0m_objective_readiness_review_before_reward_implementation"' in completed.stdout
    assert '"tau_requirement_min": 0.9' in completed.stdout
    assert '"stage5_1_plan_only_allowed": true' in completed.stdout
    assert '"recommended_next_task": "stage_5_1_reward_implementation_plan_without_code"' in completed.stdout
    assert '"reward_code_allowed": false' in completed.stdout
    assert '"training_allowed": false' in completed.stdout
    assert '"final_tau_selected": false' in completed.stdout
    assert '"all_readiness_gates_passed": true' in completed.stdout


def test_stage5_0m_source_avoids_sampling_model_code_v5_and_legacy_metrics() -> None:
    source = _read(
        ROOT
        / "src"
        / "marl_topology"
        / "evaluation"
        / "objective_readiness_review.py"
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
        "P_eff_soft",
        "P_eff_hard",
        "hard_eval",
        "soft_train",
        "D:\\PhD_works\\v5",
    ]
    hits = [term for term in banned_terms if term in source]
    assert not hits, f"Stage 5.0m source added forbidden terms: {hits}"
