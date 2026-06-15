import ast
from pathlib import Path

from marl_topology.evaluation.stage26_health_diagnostics import (
    build_stage26_full_system_health_report,
    build_stage26_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
STAGE26_SOURCE_PATHS = (
    ROOT / "src" / "marl_topology" / "evaluation" / "stage26_health_diagnostics.py",
    ROOT / "scripts" / "replay" / "stage26_full_system_health_report.py",
)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def test_stage26_manifest_is_report_only_and_blocks_training_artifacts() -> None:
    manifest = build_stage26_manifest()

    assert manifest["stage_id"] == "stage_26_full_system_health_diagnostic"
    assert manifest["config_id"] == "stage26_no_training_full_system_health_diagnostic_v1"
    assert manifest["artifact_write_allowed"] == "manifest_validated_reports_only"
    assert manifest["checkpoint_creation_allowed"] is False
    assert manifest["training_scale_up_allowed"] is False


def test_stage26_report_records_no_forbidden_training_or_tuning_actions() -> None:
    report = build_stage26_full_system_health_report(project_root=ROOT)

    assert report["pass_gate"] is True
    assert report["diagnostic_scope"]["training_updates_run"] is False
    assert report["diagnostic_scope"]["active_stack_changed"] is False
    assert report["forbidden_action_flags"] == {
        "new_training_updates_run": False,
        "scale_up_training_run": False,
        "reward_weights_changed": False,
        "hyperparameter_tuning_performed": False,
        "sampler_switched": False,
        "checkpoint_created": False,
        "legacy_v5_modified": False,
    }


def test_stage26_sources_do_not_call_training_updates_or_checkpoint_io() -> None:
    forbidden_calls = {
        "backward",
        "step",
        "zero_grad",
        "torch.save",
        "torch.load",
        "run_stage25_small_scale_pilot",
        "run_stage25_training",
        "train",
        "fit",
    }
    hits: list[str] = []

    for path in STAGE26_SOURCE_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(ROOT).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _call_name(node.func)
                lowered = name.lower()
                if name in {"torch.save", "torch.load"}:
                    hits.append(f"{relative}:{node.lineno}:{name}")
                if lowered.rsplit(".", maxsplit=1)[-1] in forbidden_calls:
                    hits.append(f"{relative}:{node.lineno}:{name}")

    assert not hits, f"Stage 26 source calls forbidden training/checkpoint APIs: {hits}"
