# Stage 22 - Objective-Aware Teacher

The Stage 22 teacher replaces projected-greedy-only supervision with
objective-aware projected search.

## Objective Order

The teacher ranks candidate projected topologies by:

1. feasibility under `tau_requirement_min = 0.9`;
2. lower latency;
3. lower energy;
4. fewer selected edges as a diagnostic tie-breaker.

Teacher candidates are projected through the same option-specific assembler
before selection. This keeps teacher-selected actor targets aligned with the
deployment projection layer.

## Target Contract

Actor targets contain:

- `actor_edge_utility_target`;
- `actor_edge_utility_confidence`;
- ranking pairs;
- low-priority reasons;
- teacher selected/rejected diagnostics;
- teacher feasibility.

Critic-only targets contain global deltas such as
`delta_consensus_success_probability`, `delta_latency`, `delta_energy`,
`delta_feasibility`, `oracle_membership`, and `objective_value`.

Actor inputs and actor targets do not include global topology, oracle labels,
consensus probability, latency, energy, reward surrogate, future outcome, or
global edge-delta targets.

## Result

For `undirected_physical_link_v1`, the projected objective-aware teacher
reached tau-feasible rate `0.7`, with mean selected edge count `3.0`. The
target distribution had `12` high, `23` mid, and `107` low-priority examples
with `44` ranking pairs.

For `directed_outgoing_v1`, projected directed teacher feasibility was `0.0`,
which contributed to archiving that option.

policy-gradient was not run in Stage 22.
