# T4 — experiment plan (edge-level recurrent state; task 3.4)

## Hypothesis (one variable: the recurrence LOCUS)
The CSI dynamics are EDGE attributes (ef cols 0-3), but the actor's recurrence is a per-NODE GRUCell; edges read
it only via endpoint node states. Add an EDGE-indexed GRU carrying a per-edge hidden [E,H] across frames (edges
are a fixed candidate set per scene), with the belief head reading that edge hidden directly. Hypothesis: an
edge-level temporal state recovers the MAGNITUDE the per-node state could not (T3), beating the stale-echo floor
and/or improving the recovered-psucc anchor.

## Controlled variable
recurrence locus: node (`belief()`, per-node GRUCell — T3) vs edge (`belief_edge()`, per-edge GRUCell — T4).
Both use the CORRECTION parametrization (belief_logit = stale_logit + head) + residual_leak=0.1, same features,
loss, weights, epochs, init seed. `edge_recurrent` is opt-in (default off, byte-identical).

## Metrics (5 seeds × {random,urban}, delay-1, held; 95% CI)
- `edge_beats_floor` / `node_beats_floor` = stale_echo_MSE − belief_MSE.
- `edge_vs_node_mse` = node_MSE − edge_MSE (does the edge locus reduce error?).
- `dir_acc` (edge / node).
- `feas_gain` (edge / node) = feas(recovered → anchor) − feas(stale → anchor).

## Load-bearing tests
`test_temporal_recovery_T4_edge_recurrence.py`: edge_recurrent adds an edge GRU (2 recurrent modules) /
belief_edge returns a per-edge hidden and carrying it changes the belief (load-bearing) / belief_edge requires
the flag / edge modules appended last preserve the node+belief_head init (back-compat).

## Decision rule
- edge beats the floor and/or beats node (CI>0) → the edge locus recovers magnitude → KEEP+adopt.
- edge ≈ node ≈ floor (spans 0) → NEGATIVE (locus is not the fix); keep opt-in, proceed to T5 (uncertainty).

## Definition of done
Failing-first tests; 5-seed CIs + raw artifact; scope explicit; decision.md; adversarial Workflow;
edge_recurrent=False byte-identical (full prior suite green).
