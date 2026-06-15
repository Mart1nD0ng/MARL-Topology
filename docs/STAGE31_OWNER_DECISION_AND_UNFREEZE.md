# Stage 31 Owner Decision, Contract Unfreeze, and tau-Feasibility Strategy

This document records the owner decision that authorizes the Stage 31
production-readiness work, the specific contract slots that are unfrozen, the
strategy for making `tau_requirement_min = 0.9` reachable without faking
feasibility, and the phase plan that follows.

It supersedes the `post_stage30_repair_loop_blocked_awaiting_owner_decision`
hold. It does **not** authorize claiming production-training readiness; that
claim is reserved for the Stage 31 Phase F readiness report after the
large-scale test runs.

## 1. Owner Decision (recorded)

The owner reviewed the Stage 26-30 root-cause analysis and the deep review and
decided:

1. `tau_requirement_min = 0.9` is **maintained**. Rationale: a planned topology
   whose consensus success probability is below 0.9 has no practical value, so
   the constraint is a real requirement, not a tunable. The job is to make 0.9
   *reachable* by strengthening scenario physics and data, not to lower it.
2. The necessary contract slots are **unfrozen** (see Section 2).
3. The Stage 30 B1 reward-objective repair (feasibility-first surrogate) is
   **approved** for active implementation.
4. The mean-field expected-initiator PBFT reliability model is **kept**. No
   message-level PBFT simulator is required.
5. The three-step plan (unfreeze+tau -> reward+actor architecture -> data) is
   approved for full code implementation.
6. After all phases complete, run a large-scale test and produce a
   comprehensive production-training-readiness report.

`owner_decision_required: false` for Stage 31 execution. `owner_approval_id:
owner_approved_stage31_production_readiness_unfreeze`.

## 2. Unfrozen Contract Slots

The following previously frozen slots are unfrozen for Stage 31, within the
boundaries stated:

| Slot | Previous state | Stage 31 authorization | Boundary kept |
| --- | --- | --- | --- |
| Reward surrogate structure | frozen flat weighted sum (`surrogate_interface_stage5_2`) | add and activate a feasibility-first structure | reliability stays a constraint; latency/energy stay objectives; `tau=0.9` fixed |
| Reward weight / normalization calibration | blocked | recalibrate references and weights on a feasible scenario set | no reward hacking; metric governance unchanged |
| Actor feature schema | frozen `STAGE18_FEATURE_SCHEMA_ID` actor-safe set | add actor-safe constraint-state fields (tx/rx budget, endpoint occupancy, conflict group) | Dec-POMDP locality preserved; no oracle/global leakage |
| Action semantics / sampler | frozen top-k Plackett-Luce only | activate a budget-aware proposal sampler and align logprob with the projected topology | single active sampler; no actor leakage |
| Scenario fixture set | frozen 3 active + ~10 stage16 fixtures | add a procedural scenario generator producing many leakage-split contexts | physics regime declared; mean-field PBFT kept |
| Label source | exhaustive oracle capped at 10 edges | add a scalable heuristic teacher for larger graphs; keep exact oracle on small holdout | oracle label never an actor input |
| Train/eval/test split | `all_rows_no_train_split` | add a context-keyed leakage-checked split | no train/eval context overlap |

Still blocked (unchanged): lowering final tau below 0.9, treating full graph as
oracle, oracle labels as deployment actor inputs, and v5 code migration.

## 3. Why tau=0.9 is currently unreachable (root)

In the production objective stack
(`src/marl_topology/data/stage21_objective_stack_evidence.py`),
`consensus_success_probability` is the expected-initiator PBFT reliability
computed over Stage 3 directed network records. Each record's per-link
reliability is the finite-blocklength `deadline_delivery_probability`, which is
driven by the channel SINR (distance, tx power, bandwidth) and the link/deadline
budget. PBFT reliability is then a quorum/fault-tolerance aggregate of those
per-link reliabilities.

The blocking fact: the existing scenario fixtures place follower nodes tens to
hundreds of meters from the leader, and several fixtures were authored against
the old Stage 2 consensus boolean (`success_probability_threshold = 0.45`), not
the 0.9 objective. With those geometries the strong links cannot form a
connected quorum-spanning core whose **minimum** PBFT-relevant reliability stays
above 0.9, so most operating points sit below the constraint (mean margin
~`-0.296`, ~40% of slots infeasible).

## 4. tau-Feasibility Strategy (no faking)

The legitimate physical levers in the production stack are, in order of effect:

1. **Node geometry** (`Scene3D` positions): place a connected core of nodes that
   are close enough to the leader and to each other that their finite-blocklength
   links are high-reliability.
2. **tx power** (`ChannelModelConfig.default_tx_power_dbm`): the radio/RSU
   quality lever already swept in Stage 5.0l.
3. **bandwidth** (`ChannelModelConfig.bandwidth_hz`, `LinkTransmissionConfig`):
   raises finite-blocklength rate margin.
4. **deadline / payload** (`LinkTransmissionConfig.deadline_s`, `payload_bits`):
   the deadline budget for retransmission.
5. **interference / resource orthogonality**: orthogonal resources remove
   cross-link interference penalties.

**Design principle — feasibility gradient, not saturation.** Stage 31 scenario
generation must produce a genuine gradient:

- A *good* topology (a sparse, strong-link, quorum-spanning core) reaches
  `consensus_success_probability >= 0.9`.
- A *naive dense* topology (full graph including far/weak edges) is dragged below
  0.9 by its weakest PBFT-relevant link, so it is infeasible or resource-dominated.
- A meaningful fraction of generated scenarios remain genuinely infeasible (no
  strong quorum-spanning core exists).

Concretely this means **NOT** inflating all link probabilities to ~1.0. The
acceptance check for the generator (Phase B) is a feasibility distribution where
the feasible-by-good-topology fraction and the infeasible fraction are both
materially non-zero, and where the full graph is frequently *not* the best
topology. Saturated all-feasible or all-infeasible generators are rejected.

## 5. Phase Plan

- **Phase A** (this document): owner decision, unfreeze, tau strategy, state update.
- **Phase B**: procedural scenario generator that produces a validated
  feasibility gradient at scale through the existing finite-blocklength +
  expected-initiator PBFT stack.
- **Phase C**: feasibility-first reward surrogate (B1), recalibrated references
  and weights on the feasible scenario set, with contract tests proving
  objective-rank vs reward-rank alignment.
- **Phase D**: actor-safe constraint-state features (B3) and a budget-aware
  sampler with projection-consistent logprob, proven to reduce
  `tx_budget_exceeded` projection rejection.
- **Phase E**: scalable heuristic teacher labels for larger graphs, a
  context-keyed leakage-checked train/eval/test split, and a regenerated
  production-scale evidence dataset with manifest and quality report.
- **Phase F**: a large-scale MAPPO readiness test on the new data with the fixed
  surrogate and fixed actor, a re-run of the 11-component health diagnostic and
  scale-readiness scorecard, and a comprehensive production-training-readiness
  report.

## 6. Boundaries that remain in force

- `tau_requirement_min = 0.9` stays fixed everywhere.
- Mean-field expected-initiator PBFT reliability is kept; no message-level sim.
- Metric governance is unchanged; no new objective metric names.
- Dec-POMDP deployment locality is preserved; no oracle or global counterfactual
  leakage into actor inputs.
- No v5 code migration.
- Production-training readiness is asserted only by the Phase F report, not by
  any earlier phase.
