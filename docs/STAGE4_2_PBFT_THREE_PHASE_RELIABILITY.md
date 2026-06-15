# Stage 4.2 / Stage 4.4 PBFT Reliability Records

Stage 4.2 PBFT Three-Phase Reliability Record remains the primary-specific
helper. Stage 4.4 adds topology-level expected initiator reliability.

## Controlled Object

The controlled object is the Stage 4 PBFT reliability module:

`src/marl_topology/protocol/pbft_reliability.py`

It consumes declared message-delivery matrices and returns analytic reliability
records.

## Desired State

Stage 4.2 should provide the smallest useful PBFT reliability evaluator after
Stage 4.1:

- explicit `pre_prepare`, `prepare`, and `commit` phases;
- explicit `n >= 3f + 1` validation;
- explicit `2f + 1` total quorum and `2f` external quorum when self-message is
  counted;
- heterogeneous message probabilities;
- conservative unknown-fault filtering option;
- no Stage 3 adapter;
- no reward, timeout gate, latency, energy, training, or v5 code migration.

In short: no Stage 3 adapter inside the fixed-primary helper, no reward, and no
v5 code migration.

Boundary shorthand: no Stage 3 adapter, no reward, no v5 code migration.

## Implemented Interfaces

- `PBFT_RELIABILITY_VARIANT_ID`
- `PBFT_EXPECTED_INITIATOR_MODEL_ID`
- `PBFT_PHASE_NAMES`
- `FAULT_FILTER_NONE`
- `FAULT_FILTER_REMOVE_LARGEST`
- `PBFTThreePhaseConfig`
- `PBFTThreePhaseReliabilityRecord`
- `PBFTExpectedInitiatorConfig`
- `PBFTExpectedInitiatorReliabilityRecord`
- `evaluate_pbft_three_phase_reliability`
- `evaluate_pbft_given_primary`
- `evaluate_expected_initiator_pbft_reliability`

Active protocol variant:

`stage4_pbft_three_phase_closed_form_v0`

Topology-level expected-initiator model:

`pbft_expected_initiator_mean_field_v1`

## Inputs

- `node_ids`
- `primary_id`
- `fault_tolerance`
- `fault_filter_mode`
- `pre_prepare_matrix`
- `prepare_matrix`
- `commit_matrix`

Each matrix is a mapping from `(sender_id, receiver_id)` to message delivery
probability in `[0, 1]`. Missing directed entries are treated as zero delivery.
Self-message entries are rejected.

## Primary-Specific Formula

The fixed-primary evaluator is now treated as a primary-specific helper. It is
not the topology-level consensus reliability by itself.

Pre-prepare readiness:

```text
alpha_1[primary] = 1
alpha_1[i] = M_pre_prepare[primary, i] for i != primary
```

Prepare cascade:

```text
prepare_inputs_i[j] = alpha_1[j] * M_prepare[j, i], j != i
alpha_2[i] = alpha_1[i] * H_ge_{2f}(filtered prepare_inputs_i)
```

Commit cascade:

```text
commit_inputs_i[j] = alpha_2[j] * M_commit[j, i], j != i
alpha_3[i] = alpha_2[i] * H_ge_{2f}(filtered commit_inputs_i)
```

Round reliability:

```text
consensus_success_probability = H_ge_{2f + 1}(filtered alpha_3)
```

`H_ge_k` is the Stage 4.1 heterogeneous quorum-tail utility.

## Stage 4.4 Expected-Initiator Formula

Stage 4.4 defines topology-level expected initiator reliability:

```text
R_consensus(G, s) = (1 / |V|) * sum_p Psi_p(G, s)
```

`Psi_p` is the primary-specific three-phase reliability with node `p` acting as
the PBFT primary. Default configuration:

- `initiator_as_primary = true`
- `primary_distribution = uniform`
- `self_vote_counted = true`
- `fault_filter = none / remove_largest`
- `model_id = pbft_expected_initiator_mean_field_v1`
- `mean_field_assumption = true`
- `view_change_mode = deferred`

The returned record contains:

- `per_primary_reliability`
- `primary_distribution`
- `consensus_success_probability`
- `model_id`
- explicit mean-field and view-change flags

`fault_filter_remove_largest` remains a conservative engineering lower-bound
approximation. It is not a strict Byzantine adversary model.

## Output Record

`PBFTThreePhaseReliabilityRecord` contains:

- protocol variant;
- node ids and primary id;
- fault tolerance and fault filter mode;
- total and external quorum;
- phase names;
- `pre_prepare_readiness`;
- `prepared_probability`;
- `committed_probability`;
- `consensus_success_probability`;
- boundary flags stating that Stage 4.2 uses declared matrices only, keeps the
  mean-field assumption explicit, and does not use a Stage 3 adapter.

`PBFTExpectedInitiatorReliabilityRecord` additionally contains per-primary
values and the uniform primary distribution.

## Metric Governance

Stage 4 maps its analytic reliability to the existing
metric-governance concept:

`consensus_success_probability`

It does not create a new metric name and does not emit CSV rows. Latency,
energy, timeout, quorum count, and reward remain separate concepts.

## Tests

Implemented tests cover:

- invalid `n < 3f + 1`;
- invalid primary and fault tolerance;
- all-one matrices produce probability one for `n = 4, f = 1`;
- zero prepare or zero commit phase prevents consensus;
- improving a message probability does not reduce reliability;
- conservative fault filtering is no larger than no filtering;
- missing directed entries are zero delivery;
- invalid self-message, unknown endpoint, and out-of-range probability entries;
- source scan rejects subset enumeration, random sampling, Monte Carlo routes,
  v5 imports, torch, training, actor/critic, COMA, and old metric aliases.
- symmetric complete graph gives equal reliability for all primaries;
- weak-primary cases lower only that primary-specific `Psi_p` and the topology
  reliability is the uniform average;
- improving one primary's outgoing links improves that primary and the average;
- fixed-primary helper matches the old fixed-primary path;
- conservative filtering lowers or preserves expected reliability.

## Boundary

Stage 4.2 does not:

- build message matrices from Stage 3 communication records;
- apply phase latency budgets;
- compute protocol latency or energy;
- define reward;
- train a model;
- implement actor, critic, COMA, GNN, or LSTM;
- copy v5 code.

Stage 4.4 additionally does not:

- report a single global fixed primary as topology reliability;
- implement view-change;
- implement a strict Byzantine adversary model;
- use Monte Carlo, random sampling, or subset enumeration.

## Residual Risks

- The formula is mean-field and does not model all shared-message correlations.
- The current evaluator is probability-space; larger committees or extreme
  probabilities may need Stage 4.1a log-space review.
- View change, leader failure, equivocation, known-fault fixtures, and
  adversarial timing are still deferred.
- Stage 4.3 must carefully prevent Stage 3 `network_delivery_probability` from
  being renamed directly as PBFT consensus reliability.
