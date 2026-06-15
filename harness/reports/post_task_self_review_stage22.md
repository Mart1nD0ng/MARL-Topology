# Stage 22 Post-Task Self-Review

Completed task: Stage 22 action semantics A/B trial, objective-aware teacher,
full message-passing GNN repair, supervised MLP/GNN rerun, fair evaluation,
selection, cleanup, and closeout.

Intended desired state: compare `directed_outgoing_v1` and
`undirected_physical_link_v1` under final Stage 3/4 objective evidence, use
objective-aware projected teachers, replace toy GNN with a real local
message-passing GNN, select one active semantics, archive the loser, and keep
policy-gradient blocked.

Actual achieved state: `undirected_physical_link_v1` was selected and promoted
as the only active action semantics. `directed_outgoing_v1` is archived as
inactive Stage 22 evidence. `LocalGNNEdgeScorer` now uses full local
message-passing and the active model id is
`local_message_passing_gnn_edge_scorer_v2`.

Evidence:

- selected semantics: `undirected_physical_link_v1`;
- losing semantics: `directed_outgoing_v1`;
- selected full-GNN projected tau-feasible rate: `0.7`;
- directed full-GNN projected tau-feasible rate: `0.0`;
- selected objective-aware teacher projected tau-feasible rate: `0.7`;
- selected target distribution: `12` high, `23` mid, `107` low, `44` ranking pairs;
- Stage 20 metrics were historical diagnostics only.

Tests and gates passed:

- action-semantics A/B behavior tests;
- objective-aware teacher target tests;
- full GNN message-passing tests;
- low-entropy cleanup contract;
- no-policy-gradient contract;
- Stage 22 hard gates in the report.

Gates deferred:

- policy-gradient readiness review is deferred to owner-approved Stage 23;
- scale-up training is deferred;
- COMA and Transformer remain deferred.

New risks:

- Stage 22 is still small-fixture supervised evidence, not statistical
  scale-up evidence.
- Directed semantics may need a separate future design if directed routing
  becomes a required domain objective, but it is not active.
- The selected full GNN matches teacher feasibility on this fixture set but has
  higher latency/energy than the projected greedy baseline.

Regressions protected:

- actor inputs remain local-only;
- full graph remains a baseline, not an oracle;
- raw baselines are diagnostic only;
- policy-gradient, COMA, Transformer, reward tuning, final tau selection,
  checkpoint writes, and `v5` modification remain blocked.

Candidate next tasks:

- `stage_23_controlled_policy_gradient_pilot_readiness_review`;
- `stage_23_selected_physical_semantics_supervised_calibration`;
- `stage_23_projection_resource_diagnostic_hardening`.

Recommended next task:

`stage_23_controlled_policy_gradient_pilot_readiness_review`

Owner decision required: true.

Required answers:

1. Which action semantics won, and why? `undirected_physical_link_v1`, because
   it aligned with physical topology evaluation and reached full-GNN projected
   tau-feasible rate `0.7` while directed reached `0.0`.
2. Was the loser removed from active code? Yes. `directed_outgoing_v1` is
   archived inactive and is absent from the active registry.
3. Is the selected semantics aligned with evaluator topology semantics? Yes.
   The selected action is physical link activation consumed as selected
   physical edges by the evaluator.
4. Is the full GNN actually message-passing? Yes. It uses node/edge encoders,
   edge-to-node messages, node-to-edge updates, two message-passing layers, and
   permutation-invariant aggregation.
5. Was toy GNN removed or replaced? Replaced. The public import remains, but
   the active implementation/model id is the full message-passing GNN.
6. Did objective-aware teacher improve over projected greedy-only teacher? It
   changed the teacher contract from reliability-only to feasibility, latency,
   energy, and sparsity after projection; selected teacher tau-feasible rate is
   `0.7`.
7. Are baselines fair under the same assembler? Yes. Main comparisons use
   projected baselines under the option-specific assembler.
8. Is policy-gradient still blocked? Yes. Stage 22 did not run policy-gradient;
   Stage 23 owner approval is required.
9. What exact Stage 23 task is recommended?
   `stage_23_controlled_policy_gradient_pilot_readiness_review`.
