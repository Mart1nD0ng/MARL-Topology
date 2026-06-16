"""Tests for the decentralized CTDE multi-agent flow (Task 3: genuine MARL, non-bandit).

Headline guarantees:
- the loop runs end-to-end on real generated scenes and produces finite MARL losses;
- it is genuinely SEQUENTIAL, not a bandit: an action depletes a node's energy battery,
  which shrinks that node's feasible action set at the NEXT step (action-consequence state);
- the actor is decentralized (no global topology) and the critic is training-only (CTDE).
"""

import pytest
import torch

from marl_topology.training.decentralized_marl import (
    DECENTRALIZED_CTDE_FLOW_ID,
    PRODUCTION_ACTOR_EDGE_DIM,
    PRODUCTION_ACTOR_NODE_DIM,
    DecentralizedCTDEFlow,
    DecentralizedMARLConfig,
    build_ctde_critic,
    build_production_actor,
    run_decentralized_marl_training,
)


def test_production_actor_and_critic_dims() -> None:
    config = DecentralizedMARLConfig()
    actor = build_production_actor(config)
    critic = build_ctde_critic(config)
    assert actor.config.node_input_dim == PRODUCTION_ACTOR_NODE_DIM == 9
    assert actor.config.edge_input_dim == PRODUCTION_ACTOR_EDGE_DIM == 12
    assert critic.config.node_feature_dim == 9
    assert critic.config.edge_feature_dim == 12
    assert actor.boundary_report()["global_topology_used"] is False
    assert critic.boundary_report()["training_only"] is True
    assert critic.boundary_report()["deployment_actor_receives_critic_output"] is False


def test_energy_battery_makes_it_sequential_not_bandit() -> None:
    # The core non-bandit property: an action at step t spends battery, so the SAME node's
    # feasible action set at step t+1 is strictly smaller -- a state transition the agent
    # controls. A bandit would have an identical action set every step regardless of action.
    config = DecentralizedMARLConfig(energy_unit_j=1.0, energy_budget_j=2.0)
    flow = DecentralizedCTDEFlow(config)
    radio = {"v0": 2, "v1": 2, "rsu0": 64, "rsu1": 64}
    energy = {"v0": 2.0, "v1": 2.0, "rsu0": 64.0, "rsu1": 64.0}

    budgets_before = flow._effective_budgets(energy, radio)
    assert budgets_before["v0"] == 2  # full battery -> full radio budget

    # v0 activates two incident links this step -> spends its whole battery.
    flow._spend_energy(energy, ("rsu0--v0", "v0--v1"))
    assert energy["v0"] == 0.0

    budgets_after = flow._effective_budgets(energy, radio)
    assert budgets_after["v0"] == 0  # depleted -> cannot transmit next step
    assert budgets_after["rsu1"] == 64  # a node not incident to any chosen edge is unaffected
    assert budgets_after["v0"] < budgets_before["v0"], "action did not change the next-step state"


def test_flow_id_and_config_freeze() -> None:
    assert DecentralizedCTDEFlow(DecentralizedMARLConfig()).flow_id == DECENTRALIZED_CTDE_FLOW_ID
    with pytest.raises(ValueError):
        DecentralizedMARLConfig(tau_requirement_min=0.8)


def _small_specs(count: int = 4):
    from marl_topology.data.stage31_scenario_generator import (
        ProductionScenarioConfig,
        generate_production_scenarios,
    )

    specs = generate_production_scenarios(
        ProductionScenarioConfig(seed=7, scenario_count=count, node_count_choices=(4, 5))
    )
    return list(specs)


@pytest.mark.slow
def test_flow_runs_end_to_end_on_real_scenes() -> None:
    specs = _small_specs(4)
    train_specs, eval_specs = specs[:3], specs[3:]
    config = DecentralizedMARLConfig(
        rounds=2,
        hidden_dim=32,
        critic_hidden_dim=32,
        rollout_steps=2,
        max_updates=1,
        update_epochs=1,
        minibatch_size=8,
        eval_every=1,
        energy_budget_j=4.0,
        seed=11,
    )
    report = run_decentralized_marl_training(config, train_specs, eval_specs)

    assert report["flow_id"] == DECENTRALIZED_CTDE_FLOW_ID
    assert report["is_decentralized_execution"] is True
    assert report["is_sequential_mdp"] is True
    assert report["is_bandit"] is False
    assert report["actor_boundary"]["global_topology_used"] is False
    assert report["critic_boundary"]["training_only"] is True
    assert "final_eval" in report
    assert 0.0 <= report["final_eval"]["tau_feasible_rate"] <= 1.0
    # At least one update must have produced finite MARL losses.
    finite_updates = [
        record for record in report["update_metrics"] if "total_loss" in record
    ]
    assert finite_updates, "no MARL update produced a loss"
    for record in finite_updates:
        assert record["total_loss"] == record["total_loss"]  # not NaN
        assert record["approx_kl"] >= 0.0
        assert 0.0 <= record["clip_fraction"] <= 1.0


@pytest.mark.slow
def test_parallel_worker_functions_run_directly() -> None:
    # Verify the multiprocessing worker functions are correct by calling them DIRECTLY
    # (serial, no process Pool). The Pool path itself is exercised on the deployment Linux
    # box; multiprocessing-torch is not stress-tested on Windows by design.
    from marl_topology.training.decentralized_marl import (
        _bc_worker,
        _rollout_worker,
        build_ctde_critic,
        build_production_actor,
    )

    specs = _small_specs(2)
    config = DecentralizedMARLConfig(
        rounds=2, hidden_dim=32, critic_hidden_dim=32, rollout_steps=2,
        energy_budget_j=6.0, bc_teacher_sa_iters=20, bc_teacher_restarts=2, seed=3,
    )
    actor = build_production_actor(config)
    critic = build_ctde_critic(config)
    actor_state = {key: value.detach().cpu() for key, value in actor.state_dict().items()}
    critic_state = {key: value.detach().cpu() for key, value in critic.state_dict().items()}

    transitions = _rollout_worker((config, actor_state, critic_state, specs[0], 0, 123, False))
    assert isinstance(transitions, list)
    if transitions:
        assert transitions[0].node_features.device.type == "cpu"

    sample = _bc_worker((config, specs[0], 0))
    assert sample is None or len(sample) == 4
