# Stage 25 Reward Surface and Objective Alignment

Stage 25 added reward-surface diagnostics as a sensor only. It did not tune or
change reward weights.

## Controlled System

- Controlled object: fixed topology/evaluation rows scored by the Stage 5
  frozen surrogate and ordered against the Stage 3/4 objective metrics.
- Desired state: the surrogate should broadly respect feasibility first,
  latency second, and energy third on comparable topology rows.
- State variables: policy label, scenario id, selected edge count, candidate
  edge count, consensus success probability, latency, energy,
  feasible-under-tau flag, surrogate reward, reliability violation component,
  latency component, energy component, and dominated/non-dominated label.
- Sensors: `build_reward_surface_analysis`, reward rank order, objective rank
  order, per-check issue lists, and the generated reward-surface CSV/JSON.
- Actuators: diagnostics and blocker classification only. Reward weights are
  not an actuator in Stage 25.
- Disturbances: incomparable scenarios, sparse/full topology trade-offs,
  empty topology edge cases, and surrogate-vs-objective disagreement.
- Coupling: the reward surrogate consumes the same objective metrics used by
  evaluation, but the scalar reward can still rank trade-offs differently from
  the pass/fail objective.

## Analysis Rows

Rows include the supervised actor topology, MAPPO actor topology, projected
greedy topology, projected full topology, objective-aware teacher topology, and
available sparse/random/empty diagnostics. Dominance and objective inversion
checks are scenario-local, so rows from unrelated scenario ids do not create
false dominance findings.

## Required Checks

The generated analysis passed all required checks:

- Feasible topology was not ranked below a clearly dominated infeasible topology
  within the same scenario.
- Above tau, the reliability violation component was zero and did not dominate
  latency/energy.
- Empty topology did not receive high reward.
- Full topology did not automatically dominate sparse feasible topology.
- Reward rank broadly aligned with objective ordering: feasibility first, then
  latency, then energy.

The report records:

- `analysis_id: stage25_reward_surface_objective_alignment_v1`
- `objective_ordering: feasibility_first_then_latency_then_energy`
- `weight_tuning_performed: false`
- `alignment_passed: true`

## Interpretation

The reward-surface sensor did not identify a hard mismatch blocker. The main
pilot still showed surrogate reward worsening while latency and energy improved,
so Stage 26 should treat reward/objective tension as residual risk rather than
as permission to tune reward weights in Stage 25.

## Verification

- `tests/unit/test_stage25_reward_surface_analysis.py`
- `result_save/stage25_small_scale_mappo_training_pilot/stage25_pilot_base_config/reward_surface_analysis.json`
- `result_save/stage25_small_scale_mappo_training_pilot/stage25_pilot_base_config/reward_surface.csv`
- `result_save/stage25_small_scale_mappo_training_pilot/stage25_pilot_base_config/reward_surface_scatter.png`
