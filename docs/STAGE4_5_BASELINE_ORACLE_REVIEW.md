# Stage 4.5 Baseline And Oracle-Candidate Review

## Controlled Object

The controlled object is the Stage 4.5 evaluation sensor:

`src/marl_topology/evaluation/stage4_baseline_oracle_review.py`

It evaluates topology baselines using Stage 3 communication records, the Stage
4.3 message-matrix adapter, and the Stage 4.4 expected-initiator PBFT
reliability model.

## Desired State

Stage 4.5 should make topology comparison observable without starting reward or
training work:

- empty, sparse, and full graph baselines are evaluated on the same scenario;
- full graph remains a baseline, not an oracle;
- a failing baseline does not prove the scenario infeasible;
- a bounded oracle-candidate search can find a feasible non-full topology when
  one exists;
- reliability feasibility is checked through registered
  `consensus_success_probability`;
- latency and energy remain registered objective quantities, not hidden reward
  terms.

## Implementation Status

Implemented public interfaces:

- `Stage45ReviewConfig`
- `Stage45TopologyReviewRow`
- `build_stage4_5_baseline_oracle_review`

Active stage id:

`stage_4_5_baseline_and_oracle_review`

Reference scenario:

`stage4_5_pbft_reference`

Protocol model:

`pbft_expected_initiator_mean_field_v1`

## Evaluation Flow

For each selected topology:

1. Build Stage 3 route records for all directed sender/receiver pairs.
2. Reuse those directed communication records for `pre_prepare`, `prepare`,
   and `commit` phase matrices.
3. Apply Stage 4.3 phase-budget gating.
4. Evaluate Stage 4.4 expected-initiator PBFT reliability.
5. Report registered metric concepts only:
   `consensus_success`, `consensus_success_probability`, `latency`, `energy`,
   and `topology_diagnostics`.

`latency` and `energy` now use Stage 4.6 protocol accounting:

- `stage4_6_sum_phase_max_clipped_to_budget`
- `stage4_6_sum_scheduled_attempt_energy`

This is a review-level accounting mode, not a reward.

## Baselines

Stage 4.5 evaluates:

- `empty`
- `sparse_star`
- `sparse_chain`
- `full`

Full graph is marked as `is_full_graph_baseline = true` and
`is_oracle_candidate = false`.

## Oracle-Candidate Rule

The oracle-candidate search is a bounded small-graph feasibility audit. It
searches topology candidates only when the candidate edge count is within the
declared cap.

Selection rule:

```text
first search by increasing edge count
keep candidates where consensus_success_probability >= reliability_threshold
exclude full graph from the oracle-candidate label
tie-break by latency, then energy, then edge-id tuple
```

This is an oracle-candidate for feasibility and sparse counterfactual review.
It is not a deployment actor input, not a reward optimizer, and not a theorem
for future physics regimes.

The report field `is_deployment_actor_input` must remain `false`.

## Boundary

Stage 4.5 does not:

- train a model;
- implement reward;
- implement actor, critic, COMA, GNN, or LSTM behavior;
- export new metric names;
- label full graph as oracle;
- claim policy failure proves infeasibility;
- modify or migrate v5 code.

The bounded topology-candidate search is separate from the PBFT reliability
calculation. PBFT reliability still uses the Stage 4.1 quorum-tail utility and
does not enumerate PBFT success subsets.

## Tests

Implemented tests cover:

- expected baselines and oracle-candidate row exist;
- empty baseline failure does not prove infeasibility;
- full graph is baseline, not oracle-candidate;
- reliability feasibility depends only on the reliability threshold;
- metric rows use registered metric names;
- source scan rejects v5, reward, training, actor/critic, COMA, MAPPO, and old
  metric alias routes.

## Residual Risks

- The reference scenario is intentionally small and deterministic.
- The oracle-candidate search optimizes sparse feasibility first; it does not
  claim latency/energy optimality across all future objectives.
- Stage 3 route aggregation is still a simple review sensor, not a full PBFT
  scheduler.
- The Stage 4.4 PBFT reliability model remains a mean-field approximation.
