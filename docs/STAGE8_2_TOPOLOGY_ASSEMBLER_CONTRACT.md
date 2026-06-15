# Stage 8.2 Topology Assembler Contract

## Controlled Object

The controlled object is the environment-side topology assembler that converts
actor directed edge scores into a feasible directed communication topology.

## Desired State

The assembler is an action feasibility projection and resource scheduling layer.
It is not a reward, not an oracle unless explicitly declared as an
oracle assembler, and not a learned neural policy.

Stage 8 deploys non-learning assembler implementations and tests. It does not
run training, implement stochastic subset sampling, create checkpoints, or
write training artifacts.

## Deployment Assembler

The deployment assembler consumes actor edge scores and declared local/resource
constraints to produce a selected directed topology.

Allowed inputs:

- actor edge scores;
- candidate edge validity;
- actor-safe local or declared resource constraints;
- role constraints;
- tx/rx capacity;
- channel/resource slots;
- conflict graph or conflict group declarations;
- previous topology only if hysteresis is explicitly enabled.

Forbidden inputs:

- `consensus_success_probability`;
- global objective values;
- oracle labels;
- edge-delta targets;
- reward surrogate;
- future outcomes;
- Stage 4 PBFT reliability result;
- centralized critic outputs.

The deployment assembler must record selected directed edges, rejected edges,
rejection reasons, pre-projection count, post-projection count, assembler id,
and diagnostics.

## Training Assembler

The training assembler may consume actor proposals/scores and exploration
metadata after owner approval for a future training task.

It must record:

- `pre_projection_proposal`;
- `post_projection_topology`;
- projection diagnostics;
- proposal source and projection source;
- any unresolved logprob semantics.

It must not use oracle labels to repair actor action. PPO logprob semantics
for projected actions are deferred and unresolved in Stage 8, because
projection can change the sampled proposal before the environment transition.

## Oracle Assembler

An oracle assembler is diagnostic only. It may use objective evaluators,
topology search, counterfactual checks, or oracle candidate generation to
produce learning targets or evaluation references.

Oracle assembler output never enters:

- actor observation;
- deployment policy input;
- deployment assembler repair metadata;
- actor export;
- communication action selection at deployment.

## Baseline And Oracle Boundaries

Full graph remains a baseline, not an oracle. It is useful as a diagnostic
comparison and can be worse than sparse feasible topologies under resource
constraints.

Fixed per-agent top-k remains a baseline, not the final policy. V5 evidence
showed that fixed threshold and fixed-selection deployment can collapse into
full or empty behavior depending on calibration. Stage 8 therefore treats
top-k as a regression baseline only.

Threshold assembly remains a baseline only. It may produce full or empty
behavior and is not recommended as final deployment logic.

## Stage 8 Non-Learning Assemblers

Stage 8 implements:

- `ThresholdAssembler` - selects score `>= threshold`; baseline only.
- `FixedTopKPerAgentAssembler` - selects top-k outgoing edges per agent;
  baseline only.
- `RoleAwareAdaptiveBudgetAssembler` - uses role-dependent local budgets;
  non-learning, not final proof of optimality.
- `ConflictAwareGreedyAssembler` - recommended Stage 8 deployment assembler;
  sorts by score and greedily adds edges only when candidate validity, role
  rules, tx capacity, rx capacity, channel conflicts, and interference
  conflict groups are satisfied.

Optional future projection designs are documented but not implemented:

- Gumbel top-k;
- Plackett-Luce subset sampling;
- sequential categorical proposal policy;
- score-cost projection with local-only resource terms.

## Interface Fields

`EdgeScoreRecord` fields:

- `agent_id`;
- `neighbor_id`;
- `edge_id`;
- `directed_edge_id`;
- `score`;
- optional `probability`;
- `score_source`;
- `time_step`.

`CandidateEdgeConstraint` fields:

- `edge_id`;
- `tx_id`;
- `rx_id`;
- `edge_type`;
- `role_allowed`;
- `channel_slot`;
- `conflict_group`;
- `tx_capacity_cost`;
- `rx_capacity_cost`;
- `valid_candidate`.

`AssemblerConfig` fields:

- `assembler_id`;
- `mode`;
- optional `max_outgoing_per_agent`;
- `role_budget_config`;
- `tx_capacity`;
- `rx_capacity`;
- `conflict_policy`;
- `allow_hysteresis`;
- `deterministic`.

`AssembledTopology` fields:

- `selected_directed_edges`;
- `rejected_edges`;
- `rejection_reasons`;
- `pre_projection_edge_count`;
- `post_projection_edge_count`;
- `assembler_id`;
- `diagnostics`.

Rejection reasons:

- `invalid_candidate`;
- `role_forbidden`;
- `low_score`;
- `tx_budget_exceeded`;
- `rx_capacity_exceeded`;
- `channel_conflict`;
- `interference_conflict`;
- `duplicate_edge`;
- `hysteresis_rejected`;
- `projection_limit`.

## Coupling Map

Actor scores are coupled to topology assembly only through directed edge score
records. Assembler output is coupled to Stage 3 communication simulation as
scheduled communication candidates. Stage 4 consensus is coupled only after
Stage 3 produces delivery/message matrices.

Critic heads are coupled to future training diagnostics only. They are not
deployment assembler inputs.

Oracle outputs are coupled to learning target generation and diagnostics only.
They are not actor inputs or deployment repairs.

## Acceptance Criteria

- Deployment assembler rejects forbidden objective, oracle, reward, future, and
  PBFT reliability metadata.
- Training assembler semantics document pre-projection and post-projection
  recording requirements.
- Oracle assembler is explicitly diagnostic-only.
- Full graph is not labelled oracle.
- Fixed top-k is baseline only.
- Conflict-aware greedy is the Stage 8 recommended deployment assembler.
- No stochastic sampler is implemented in Stage 8.

## Verification Commands

```powershell
python -m pytest -q
python harness\scripts\validate_tasks.py
```

## Residual Risks

Greedy projection is deterministic and testable but not globally optimal.
Future stages may need stronger resource scheduling, differentiable projected
action accounting, or stochastic subset policies. Those changes require owner
approval and new leakage/logprob tests before training.

## Stage 31 resolution: budget-aware sampler and constraint-aware actor (owner-approved)

Stages 26-30 found that the active top-k Plackett-Luce sampler proposes the
highest-scoring edges regardless of the endpoint budget, so the deployment
assembler rejected ~100% of rejected proposals for `tx_budget_exceeded` (the B3
projection-friction blocker). Under the Stage 31 owner decision the action
semantics gain a budget-aware option:

- `physical_budget_aware_sequential_sampler`
  (`BudgetAwareSequentialProposalSampler` in
  `src/marl_topology/training/policy_gradient/samplers.py`) samples sequentially
  but only ever from edges whose both endpoints still have remaining budget. The
  proposal is therefore guaranteed endpoint-budget-feasible, so the assembler
  accepts every proposed edge and `tx_budget_exceeded` rejections drop to zero.
  Its log-probability is the exact sequential categorical log-probability over
  the eligible set at each step, so the gradient is consistent with the
  budget-feasible proposal the environment projects (closing the previously
  "unresolved" projected-action logprob gap for this sampler).
- The actor feature schema gains an additive, owner-approved
  `actor_local_edge_tensor_v2_budget_aware` variant that exposes two actor-safe,
  purely local endpoint-contention signals (`local_incident_edge_count`,
  `endpoint_contention`) so the actor is no longer blind to the constraint the
  assembler enforces. No global, budget-table, or oracle field is introduced;
  the v1 schema is unchanged.

Tests: `tests/unit/test_stage31_budget_aware_sampler.py` (zero `tx_budget_exceeded`
through the real assembler, budget-feasibility, logprob/entropy replay
consistency) and `tests/unit/test_stage31_constraint_aware_features.py`. The
historical active sampler id is unchanged; the Stage 31 sampler is selected via
`get_stage31_active_sampler()`.
