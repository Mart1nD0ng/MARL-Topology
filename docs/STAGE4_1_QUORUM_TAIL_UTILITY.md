# Stage 4.1 Heterogeneous Quorum-Tail Utility

## Controlled Object

The controlled object is the standalone Stage 4.1 quorum-tail probability
utility:

`src/marl_topology/protocol/quorum_tail.py`

It evaluates the probability that at least `k` heterogeneous Bernoulli inputs
succeed. It is a protocol utility, not a PBFT three-phase implementation.

## Desired State

The utility provides a small, testable, implementation-ready building block for
Stage 4.2 PBFT reliability records.

Desired properties:

- supports heterogeneous input probabilities;
- handles quorum boundary cases;
- is monotonic in every input probability;
- avoids success-subset enumeration;
- avoids random sampling, learned prediction, and Monte Carlo simulation;
- does not import v5 code;
- does not create reward, timeout, latency, energy, or training behavior;
- does not emit a registered metric by itself.

## Implemented Interfaces

- `QUORUM_TAIL_EVALUATOR_ID`
- `QuorumTailResult`
- `heterogeneous_quorum_tail`
- `evaluate_quorum_tail`
- `remove_largest_probabilities`
- `conservative_quorum_tail`

Active evaluator id:

`stage4_heterogeneous_quorum_tail_v1`

## Formula

For independent heterogeneous Bernoulli inputs `p_1, ..., p_m`, the target
quantity is:

```text
Pr[number of successes >= k]
```

The utility computes the generating-polynomial coefficient recurrence for:

```text
product_j ((1 - p_j) + p_j z)
```

A capped tail bucket stores probability mass for `>= k` successes, so the
implementation does not enumerate success subsets.

## Conservative Byzantine Filter

`remove_largest_probabilities(probabilities, remove_count)` removes the largest
input probabilities while preserving the original order of the remaining
values. `conservative_quorum_tail` applies this filter before evaluating the
tail probability.

This supports the Stage 4.0 plan's worst-case unknown-fault assumption without
implementing PBFT.

## Metric Governance

The result is diagnostic utility output. It is not a new project metric.

Stage 4.2 may use the utility to calculate the registered
`consensus_success_probability`, but Stage 4.1 does not produce
`consensus_success`, latency, energy, reward, or CSV rows.

## Tests

Implemented tests:

- `H_ge_0 = 1`;
- `H_ge_k = 0` when `k > m`;
- all-zero and all-one boundary cases;
- homogeneous hand formulas for three inputs;
- heterogeneous hand formula for three inputs;
- monotonicity in each input probability;
- invalid input rejection;
- conservative filtering removes the largest probabilities;
- filtered tail is no larger than unfiltered tail for the same quorum;
- source scan rejects subset enumeration, random sampling, Monte Carlo routes,
  v5 imports, and torch dependency.

## Deferred

- PBFT pre-prepare, prepare, and commit cascade;
- Stage 3 message-delivery matrix adapter;
- phase deadline gating;
- latency and energy aggregation for protocol rounds;
- reward, training, actor, critic, COMA, GNN, and LSTM work.

## Residual Risks

- The utility assumes independent Bernoulli inputs; Stage 4.2 must document the
  mean-field boundary when using it inside PBFT.
- The current arithmetic is pure float probability-space; Stage 4.2 may need a
  log-space variant if committee sizes or probabilities become extreme.
- The conservative largest-probability filter is a policy choice for unknown
  byzantine identities, not a full adversarial scheduler model.
