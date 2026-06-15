from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_stage23_removed_ab_executable_files_are_absent() -> None:
    removed = [
        ROOT / "src" / "marl_topology" / "evaluation" / "stage22_action_semantics_ab_evaluation.py",
        ROOT / "src" / "marl_topology" / "training" / "stage22_action_semantics_supervised.py",
        ROOT / "scripts" / "train" / "stage22_action_semantics_ab_full_gnn_report.py",
    ]

    assert all(not path.exists() for path in removed)
