from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_stage23_removed_ab_executable_files_are_absent() -> None:
    removed = [
        ROOT / "src" / "marl_topology" / "evaluation" / "stage22_action_semantics_ab_evaluation.py",
        ROOT / "src" / "marl_topology" / "training" / "stage22_action_semantics_supervised.py",
        ROOT / "scripts" / "train" / "stage22_action_semantics_ab_full_gnn_report.py",
    ]

    assert all(not path.exists() for path in removed)


def test_stage23_sources_do_not_run_policy_gradient_or_write_checkpoints() -> None:
    paths = [
        ROOT / "src" / "marl_topology" / "evaluation" / "stage23_policy_gradient_readiness.py",
        ROOT / "scripts" / "train" / "stage23_controlled_policy_gradient_pilot_readiness_review.py",
    ]
    banned_terms = [
        "torch.save",
        "torch.load",
        "checkpoint_path",
        "class COMA",
        "class Transformer",
        "run_controlled_policy_pilot",
        "D:\\PhD_works\\v5",
    ]
    hits = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for term in banned_terms:
            if term in text:
                hits.append(f"{path.relative_to(ROOT)}:{term}")
    assert not hits


def test_stage23_report_script_prints_json_without_artifact_write() -> None:
    text = (
        ROOT / "scripts" / "train" / "stage23_controlled_policy_gradient_pilot_readiness_review.py"
    ).read_text(encoding="utf-8")

    assert "print(json.dumps" in text
    assert "write_text(" not in text
    assert "write_bytes(" not in text
    assert "result_save" not in text
