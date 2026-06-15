from pathlib import Path

from marl_topology.training.mappo.trainer import (
    run_stage24_critic_integrated_micro_loop,
    run_stage24_preflight,
)


ROOT = Path(__file__).resolve().parents[2]


def test_stage24_preflight_confirms_selected_physical_critic_and_manifest() -> None:
    preflight = run_stage24_preflight(project_root=".")
    gates = preflight["gates"]

    assert preflight["preflight_passed"] is True
    assert gates["active_action_semantics_selected_physical"]["passed"] is True
    assert gates["discarded_action_semantics_not_active"]["passed"] is True
    assert gates["active_sampler_before_stage24_is_plackett_luce"]["passed"] is True
    assert gates["endpoint_sampler_archived_repair_candidate"]["passed"] is True
    assert gates["centralized_critic_available_training_only"]["passed"] is True
    assert gates["actor_boundary_safe"]["passed"] is True
    assert gates["reward_surrogate_config_read_only"]["passed"] is True
    assert gates["run_manifest_validator_available"]["passed"] is True
    assert gates["no_active_coma_or_transformer"]["passed"] is True


def test_stage24_actor_safe_boundary_excludes_critic_and_objective_fields() -> None:
    report = run_stage24_critic_integrated_micro_loop(mode="smoke", project_root=".")
    winner = report["sampler_selection"]["selected_sampler_id"]
    record = report["sampler_reports"][winner]["representative_records"][0]
    observation = record["actor_safe_observation"]
    critic_input = record["centralized_critic_input"]

    assert observation["view_role"] == "actor_safe_decentralized_policy_input"
    assert observation["forbidden_actor_fields_detected"] == []
    forbidden = {
        "centralized_critic_input",
        "consensus_success_probability",
        "latency",
        "energy",
        "reward_surrogate",
        "global_topology",
        "future_outcome",
        "oracle_label",
    }
    assert forbidden.isdisjoint(set(observation["field_names"]))
    assert critic_input["critic_view"]["view_role"] == "critic_centralized_training_only"
    assert critic_input["critic_view"]["actor_input_allowed"] is False


def test_stage24_sources_avoid_forbidden_legacy_and_artifact_paths() -> None:
    paths = list((ROOT / "src" / "marl_topology" / "training" / "mappo").glob("*.py"))
    paths.append(ROOT / "src" / "marl_topology" / "training" / "policy_gradient" / "samplers.py")
    banned = [
        "__import__(",
        "importlib.import_module(\"torch\")",
        "importlib.import_module('torch')",
        "class MAPPO",
        "class COMA",
        "class Transformer",
        "torch.save",
        "torch.load",
        "checkpoint_path",
        "D:\\PhD_works\\v5",
    ]
    hits = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for term in banned:
            if term in text:
                hits.append(f"{path.relative_to(ROOT)}:{term}")
    assert not hits
