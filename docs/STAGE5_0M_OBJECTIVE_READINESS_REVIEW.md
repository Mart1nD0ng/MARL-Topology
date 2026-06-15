# Stage 5.0m Objective Readiness Review Before Reward Implementation

## Scope

Stage 5.0m reviews whether the project has enough objective and feasibility
evidence to enter Stage 5.1 as a plan-only reward implementation design task.

It does not implement reward, does not choose reward weights, does not train
models, does not add actor/critic/COMA/GNN/LSTM code, does not select final
`tau_consensus`, does not lower `tau_requirement_min = 0.9`, does not migrate
v5 code, and does not treat full graph as an oracle.

Boundary shorthand: does not migrate v5 code.

## Controlled Object

The controlled object is the transition gate between:

```text
Stage 5.0 objective/reward contract and feasibility evidence
-> Stage 5.1 reward implementation plan without code
```

This is not permission to write a reward module. It is permission to design the
next implementation plan and tests.

## Desired State

The project should know whether remaining Stage 5.0 evidence gaps block a
Stage 5.1 plan-only task.

The desired state is:

- objective semantics remain stable;
- `tau_requirement_min = 0.9` remains the requirement baseline;
- Stage 3-backed feasibility levers have been observed;
- metric governance remains unchanged;
- known reward-hacking routes remain blocked;
- Stage 5.1 is recommended only as a plan without code.

## Executable Sensor

Module:

```text
src/marl_topology/evaluation/objective_readiness_review.py
```

Replay:

```powershell
python scripts\replay\stage5_0m_objective_readiness_review.py
```

The sensor composes evidence from:

- Stage 5.0h requirement-anchored feasibility diagnosis;
- Stage 5.0j minimal alpha feasibility sweep;
- Stage 5.0k Stage 3-backed bandwidth/deadline/resource sweep;
- Stage 5.0l Stage 3-backed range expansion and realism review.

## Readiness Gates

| Gate | Status | Evidence |
| --- | --- | --- |
| `tau_requirement_baseline_gate` | pass | `tau_requirement_min = 0.9` is a requirement baseline, not a fitted low threshold. |
| `objective_contract_semantics_gate` | pass | Reliability remains a constraint; latency and energy remain primary objectives. |
| `metric_governance_gate` | pass | Reports use registered metric-valued fields only and do not select final tau. |
| `failure_diagnosis_gate` | pass | Infeasible rows and families have explicit failure reasons. |
| `stage3_backed_evidence_gate` | pass | Stage 3-backed rows use finite-blocklength communication records. |
| `feasibility_lever_evidence_gate` | pass | Single-axis Stage 3-backed controls move selected rows to requirement feasibility. |
| `negative_control_gate` | pass | Reward, training, v5 migration, full-graph oracle, and actor-leakage routes remain closed. |
| `stage5_1_scope_gate` | pass | Stage 5.1 may be a plan-only task; reward code remains blocked. |
| `conservative_fault_filter_awareness_gate` | pass | `remove_largest` remains a conservative diagnostic, not a strict Byzantine model. |

## Evidence Summary

Stage 5.0h showed that at `tau_requirement_min = 0.9`, four fixture families
had at least one feasible topology and four had none. Infeasible rows were
classified by link budget, deadline, retransmission, topology candidate,
interference, PBFT quorum, primary distribution, or resource-budget causes.

Stage 5.0j showed alpha-level recovery signals for bandwidth, deadline,
resource orthogonalization, and topology candidate expansion. The conservative
fault-filter comparison stayed negative, which is expected.

Stage 5.0k grounded bandwidth, deadline, and resource orthogonalization in
Stage 3 network records, Stage 3.6 finite-blocklength links, the Stage 4.3
message-matrix adapter, Stage 4.6 accounting, and Stage 4.4 expected-initiator
PBFT reliability.

Stage 5.0l expanded Stage 3-backed coverage to tx power, payload, RSU
height/placement, and explicit resource-budget limits.

Stage 3-backed sweep coverage now includes:

- `bandwidth_sweep`
- `deadline_sweep`
- `resource_orthogonalization_sweep`
- `tx_power_sweep`
- `payload_sweep`
- `rsu_height_placement_sweep`
- `resource_budget_limit_sweep`

`fault_filter_mode_comparison` and `topology_candidate_expansion` remain useful
Stage 5.1 planning considerations, but their current evidence does not block a
plan-only Stage 5.1 task.

## Decision

Stage 5.0m decision:

```text
stage5_1_plan_only_allowed = true
reward_code_allowed = false
reward_weight_calibration_allowed = false
training_allowed = false
actor_critic_model_work_allowed = false
final_tau_selected = false
```

Recommended next task:

```text
Stage 5.1 - reward implementation plan without code
```

## Stage 5.1 Entry Conditions

Stage 5.1 should be limited to a plan-only design that specifies:

- reward-surrogate module boundary without writing code;
- use of `tau_requirement_min = 0.9` as the requirement baseline;
- reliability plateau above tau;
- latency and energy normalization references;
- reward-hacking tests;
- metric-governance tests;
- Dec-POMDP leakage tests;
- replay dataset column implications;
- explicit owner approval needed before implementation.

## Still Blocked

The following remain blocked after Stage 5.0m:

- reward implementation code;
- reward weight calibration;
- training runs;
- actor, critic, COMA, GNN, or LSTM implementation;
- final tau selection below `tau_requirement_min`;
- v5 code migration;
- oracle labels as actor inputs.

## Metric Governance

Stage 5.0m adds no new metric names.

Metric-valued fields remain:

- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

All readiness fields are diagnostics or control-state fields, not objective
metrics.

## Regression Check

Protected behavior:

- no reward implementation;
- no reward weights;
- no training;
- no model code;
- no v5 code migration;
- no lower final tau;
- no full graph as oracle;
- no oracle or sweep labels in deployment actor inputs.

## Residual Risks

- Stage 5.0 evidence is still deterministic and small-scale, not a calibrated
  city scenario distribution.
- Physical parameter realism still needs external references before deployment
  claims.
- Stage 5.1 must remain a plan unless the owner explicitly approves reward
  implementation after reviewing the plan.
