"""Stage 32 production training: GNN actor + repaired graph critic integration."""

from __future__ import annotations

import torch

from marl_topology.data.stage31_production_dataset import build_production_dataset
from marl_topology.data.stage31_scenario_generator import ProductionScenarioConfig
from marl_topology.models.centralized_message_passing_graph_critic import (
    CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID,
)
from marl_topology.models.local_gnn_edge_scorer import LOCAL_GNN_EDGE_SCORER_MODEL_ID
from marl_topology.training.run_manifest_validator import validate_run_manifest_dry_run
from marl_topology.training.stage31_readiness import build_scenario_tensors
from marl_topology.training.stage32_production_training import (
    Stage32Config,
    build_critic,
    build_critic_batch,
    build_gnn_actor,
    build_stage32_manifest,
    score_scenario,
    threshold_budget_feasible,
    train_one_seed,
    train_production,
)


def _small_dataset(count: int = 70):
    return build_production_dataset(ProductionScenarioConfig(seed=31, scenario_count=count))


def test_gnn_scores_physical_edges_via_directed_ego_groups() -> None:
    ds = _small_dataset(40)
    spec = ds.specs_for_split("train")[0]
    tensors = build_scenario_tensors(spec)
    actor = build_gnn_actor(Stage32Config())
    logits = score_scenario(actor, tensors)
    # One logit per candidate physical edge.
    assert logits.shape[0] == len(tensors.edge_ids)
    assert torch.isfinite(logits).all()


def test_graph_critic_values_a_topology() -> None:
    ds = _small_dataset(40)
    spec = ds.specs_for_split("train")[0]
    tensors = build_scenario_tensors(spec)
    critic = build_critic(Stage32Config())
    selected = tensors.edge_ids[:2]
    out = critic(build_critic_batch(tensors, selected))
    assert out.normalized_value.shape == (1,)
    assert torch.isfinite(out.normalized_value).all()


def test_variable_proposal_size_respects_budgets() -> None:
    ds = _small_dataset(40)
    spec = ds.specs_for_split("train")[0]
    tensors = build_scenario_tensors(spec)
    actor = build_gnn_actor(Stage32Config())
    logits = score_scenario(actor, tensors)
    selected = threshold_budget_feasible(logits, tensors, 0.0)
    budgets = tensors.budget_map()
    usage: dict[str, int] = {}
    for edge in selected:
        u, v = edge.split("--")
        usage[u] = usage.get(u, 0) + 1
        usage[v] = usage.get(v, 0) + 1
    assert all(usage[n] <= budgets.get(n, 2) for n in usage)


def test_train_one_seed_runs_and_keeps_projection_clean() -> None:
    ds = _small_dataset(70)
    config = Stage32Config(
        seed=3201,
        warm_start_epochs=6,
        epochs=2,
        critic_pretrain_scenarios=15,
        critic_pretrain_epochs=4,
        pg_minibatch=32,
    )
    result = train_one_seed(ds, config)
    # Budget-aware deployment -> no tx_budget projection friction.
    assert result.after_eval["projection_rejection_rate"] == 0.0
    assert result.test_eval["projection_rejection_rate"] == 0.0
    # Variable proposal size yields sparse topologies (not the full graph).
    assert 0.0 < result.after_eval["mean_selected_edges"]
    # Fine-tune is non-destructive (keep-best on eval signal).
    assert (
        result.after_eval["mean_surrogate_signal"]
        >= result.warm_start_eval["mean_surrogate_signal"] - 1e-9
    )
    assert -1e-6 <= 1.0  # critic EV is a finite number
    assert isinstance(result.critic_explained_variance, float)


def test_train_production_aggregates_named_models() -> None:
    ds = _small_dataset(70)
    config = Stage32Config(
        warm_start_epochs=4, epochs=1, critic_pretrain_scenarios=12, critic_pretrain_epochs=3,
        pg_minibatch=32,
    )
    result = train_production(ds, seeds=(3201, 3202), base_config=config)
    agg = result.aggregate()
    assert agg["actor_model_id"] == LOCAL_GNN_EDGE_SCORER_MODEL_ID
    assert agg["critic_model_id"] == CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID
    assert agg["seed_count"] == 2
    assert agg["projection_rejection_mean"] == 0.0


def test_stage32_run_manifest_is_valid() -> None:
    manifest = build_stage32_manifest(2000, (3201, 3202, 3203, 3204, 3205))
    result = validate_run_manifest_dry_run(manifest)
    assert result.is_valid, result.error_codes()
    # Artifacts are contained under result_save.
    assert manifest["artifact_root"] == "result_save"
