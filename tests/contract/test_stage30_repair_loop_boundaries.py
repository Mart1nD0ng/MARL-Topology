from pathlib import Path

from marl_topology.evaluation.stage30_repair_diagnostics import build_stage30_closed_loop_report


ROOT = Path(__file__).resolve().parents[2]
STAGE30_SOURCE = ROOT / "src" / "marl_topology" / "evaluation" / "stage30_repair_diagnostics.py"
STAGE30_REPLAY_SCRIPT = ROOT / "scripts" / "replay" / "stage30_repair_diagnostics_report.py"
STAGE30_PILOT_SCRIPT = ROOT / "scripts" / "train" / "stage30_diagnostic_mappo_pilot.py"


def test_stage30_forbidden_action_flags_are_false() -> None:
    report = build_stage30_closed_loop_report(project_root=ROOT)

    assert set(report["forbidden_action_flags"].values()) == {False}
    assert report["large_scale_readiness"]["large_scale_training_allowed"] is False


def test_stage30_sources_avoid_checkpoint_scaleup_and_architecture_patterns() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (STAGE30_SOURCE, STAGE30_REPLAY_SCRIPT, STAGE30_PILOT_SCRIPT)
    )
    forbidden = (
        "class COMA",
        "class Transformer",
        "class LSTM",
        "class GRU",
        "torch.save",
        "torch.load",
        "checkpoint_path",
        "D:\\PhD_works\\v5",
        "large_scale_training_allowed\": true",
        "tau_requirement_min = 0.8",
        "tau_requirement_min = 0.85",
    )
    hits = [term for term in forbidden if term in text]

    assert hits == []


def test_stage30_does_not_create_unmanifested_result_save_directory() -> None:
    assert not (ROOT / "result_save" / "stage30_closed_loop_repair_until_scale_readiness").exists()
    assert not (ROOT / "result_save" / "stage30_diagnostic_mappo_pilot").exists()


def test_stage30_preserves_actor_boundary_and_torch_import_hygiene() -> None:
    report = build_stage30_closed_loop_report(project_root=ROOT)
    src_text = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "src").rglob("*.py"))
    dynamic_torch_terms = ('__import__("to" + "rch")', "__import__('to' + 'rch')")

    assert report["forbidden_action_flags"]["forbidden_actor_field_added"] is False
    assert [term for term in dynamic_torch_terms if term in src_text] == []
