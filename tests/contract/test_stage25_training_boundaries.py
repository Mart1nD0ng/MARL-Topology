import ast
import json
from pathlib import Path

from marl_topology.training.mappo.stage25_pilot import (
    STAGE25_ARTIFACT_ROOT,
    STAGE25_BASE_CONFIG_ID,
    STAGE25_PASS_VERDICT,
    STAGE25_RECOMMENDED_NEXT_TASK_PASS,
    build_stage25_manifest,
)
from marl_topology.training.policy_gradient.samplers import (
    PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / STAGE25_ARTIFACT_ROOT / "training_report.json"


def _training_report() -> dict[str, object]:
    assert REPORT_PATH.exists(), "Stage 25 training report artifact is required"
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_stage25_manifest_records_fixed_stack_and_blocks_checkpoints_and_scaleup() -> None:
    manifest = build_stage25_manifest()

    assert manifest["stage_id"] == "stage_25_small_scale_formal_mappo_training_pilot"
    assert manifest["config_id"] == STAGE25_BASE_CONFIG_ID
    assert manifest["sampler_id"] == PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID
    assert manifest["active_policy_gradient_sampler_id"] == PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID
    assert manifest["action_semantics_id"] == "undirected_physical_link_v1"
    assert manifest["model_id"] == "local_message_passing_gnn_edge_scorer_v2"
    assert manifest["checkpoint_creation_allowed"] is False
    assert manifest["training_scale_up_allowed"] is False
    assert manifest["artifact_write_allowed"] == "manifest_validated_reports_only"
    assert manifest["owner_approval_id"] == "owner_approved_stage25_small_scale_formal_mappo_training_pilot"


def test_stage25_training_report_passes_formal_gate_without_forbidden_work() -> None:
    report = _training_report()
    aggregate = report["aggregate"]
    pass_fail = report["pass_fail_gate"]

    assert report["verdict"] == STAGE25_PASS_VERDICT
    assert report["pass_gate"] is True
    assert pass_fail["passed"] is True
    assert pass_fail["completed_seed_count"] >= 2
    assert aggregate["seed_count"] == 3
    assert aggregate["did_mappo_improve_supervised_on_eval"] is True
    assert aggregate["eval_delta_mean"]["tau_feasible_rate_delta"] >= -0.05
    assert aggregate["eval_delta_mean"]["violation_rate_delta"] <= 0.05
    assert (
        aggregate["eval_delta_mean"]["latency_delta"] < 0.0
        or aggregate["eval_delta_mean"]["energy_delta"] < 0.0
        or aggregate["eval_delta_mean"]["top_proposal_rejection_rate_delta"] < 0.0
        or aggregate["eval_delta_mean"]["mean_reward_surrogate_delta"] > 0.0
    )
    assert report["reward_config_unchanged"] is True
    assert report["reward_weight_tuning_performed"] is False
    assert report["diagnostic_probes_run"] == []
    assert report["primary_conclusion_based_only_on_base_config"] is True
    assert report["sampler_switched"] is False
    assert report["active_policy_gradient_sampler_id"] == PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID
    assert report["checkpoint_written"] is False
    assert report["scale_up_training_performed"] is False
    assert report["final_tau_selected"] is False
    assert report["coma_introduced"] is False
    assert report["transformer_introduced"] is False
    assert report["gru_lstm_or_recurrent_ppo_introduced"] is False
    assert report["v5_modified"] is False
    assert report["recommended_next_task"] == STAGE25_RECOMMENDED_NEXT_TASK_PASS
    assert report["owner_decision_required"] is True


def test_stage25_sources_do_not_add_forbidden_architectures_or_legacy_writes() -> None:
    paths = [
        ROOT / "src" / "marl_topology" / "training" / "mappo" / "stage25_pilot.py",
        ROOT / "src" / "marl_topology" / "evaluation" / "reward_surface_analysis.py",
        ROOT / "scripts" / "train" / "stage25_small_scale_mappo_training_pilot.py",
        ROOT / "scripts" / "replay" / "stage25_training_visualization_report.py",
    ]
    text_banned = [
        "torch.save",
        "torch.load",
        "checkpoint_path",
        "D:\\PhD_works\\v5",
    ]
    ast_banned_names = {
        "COMA",
        "Transformer",
        "LSTM",
        "GRU",
        "RecurrentPPO",
    }
    hits: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT).as_posix()
        for term in text_banned:
            if term in text:
                hits.append(f"{relative}:{term}")
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in ast_banned_names:
                hits.append(f"{relative}:class {node.name}")
            if isinstance(node, ast.FunctionDef) and node.name.lower() in {
                "coma",
                "transformer",
                "lstm",
                "gru",
                "recurrent_ppo",
            }:
                hits.append(f"{relative}:def {node.name}")

    assert not hits, f"Stage 25 introduced forbidden source patterns: {hits}"


def test_stage25_project_state_syncs_closeout_and_blocks_scaleup() -> None:
    text = (ROOT / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")

    required = [
        "current_stage: post_stage_25_complete_small_scale_mappo_pilot_awaiting_owner_decision",
        "stage_25_small_scale_formal_mappo_training_pilot",
        "stage25_small_scale_mappo_pilot_gate",
        "stage25_reward_surface_alignment_gate",
        "stage25_visualization_report_gate",
        "stage25_torch_import_hygiene_gate",
        "recommended_next_task: stage_26_scale_readiness_and_failure_mode_review",
        "owner_decision_required: true",
        "Scale-up training remains blocked",
    ]
    missing = [term for term in required if term not in text]
    assert not missing, f"PROJECT_STATE missing Stage 25 closeout terms: {missing}"
