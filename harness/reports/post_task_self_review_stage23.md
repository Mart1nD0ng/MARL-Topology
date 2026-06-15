# Post-Task Self-Review - Stage 23 Selected-Physical Policy-Gradient Landing

Completed task: implemented and ran the selected-physical policy-gradient
micro-pilot for `undirected_physical_link_v1`.

Intended desired state: build the selected-physical PG harness, compare sampler
A/B/C, use the existing Stage 5 surrogate without weight changes, run a
controlled micro-pilot, promote one low-entropy active sampler, and keep scale-up
blocked pending owner decision.

Actual achieved state: Stage 23 PASS. `physical_plackett_luce_top_k_sampler`
was promoted as the only active policy-gradient sampler. The Bernoulli and
endpoint-budgeted trial samplers are excluded from the active registry.

Evidence:

- `python scripts\train\stage23_selected_physical_policy_gradient_pilot.py`
  returned `stage23_pass_pg_pilot_landed_sampler_promoted`.
- All three samplers ran on selected physical-link semantics.
- Reward config fingerprint was unchanged before and after the pilot.
- Active sampler registry count is exactly one.

Tests and gates passed:

- selected physical-link action semantics;
- full GNN v2 actor boundary;
- actor-safe input boundary;
- physical-link assembler;
- Stage 3/4 objective evaluator;
- sampler A/B/C trial;
- low-entropy winner promotion;
- reward no-weight-change rule;
- no checkpoint or uncontrolled artifact.

Gates deferred:

- scale-up training;
- final tau selection;
- checkpoint creation;
- broader policy-gradient readiness.

New risks:

- The micro-pilot did not improve tau-feasible rate, latency, or energy for the
  winning sampler.
- Evidence is small fixed-seed evidence and should not be treated as scale-up
  readiness.
- Endpoint-budgeted sampler had better mean reward but approximate aggregate
  proposal logprob semantics and slightly higher projection mismatch.

Regressions checked:

- actor inputs still exclude consensus probability, latency, energy, reward
  surrogate, future outcome, oracle labels, and global topology;
- reward weights stayed frozen;
- losing samplers are not active;
- full graph and fixed top-k remain baselines, not final policy.

Candidate next tasks:

- stage_24_policy_gradient_pilot_analysis_and_scale_readiness_review;
- sampler logprob repair only if owner wants to revisit endpoint-local sampler;
- evidence expansion only after Stage 24 review.

Recommended next task:
`stage_24_policy_gradient_pilot_analysis_and_scale_readiness_review`.

Owner decision required: yes.

Required questions:

1. Which samplers were tested? `physical_bernoulli_proposal_sampler`,
   `physical_plackett_luce_top_k_sampler`, and
   `endpoint_budgeted_physical_proposal_sampler`.
2. Which sampler won? `physical_plackett_luce_top_k_sampler`.
3. Were losing samplers removed from active code? Yes, they are absent from the
   active sampler registry.
4. Did policy-gradient improve, preserve, or harm tau_feasible_rate? It
   preserved the winning sampler's tau-feasible rate at 0.3333.
5. Did latency/energy improve? No; they were preserved in the bounded pilot.
6. Did projection mismatch improve? No; it was preserved at 0.1111 for the
   winning sampler.
7. Did any collapse occur? Bernoulli showed empty-graph collapse risk and was
   not promoted. The winning sampler did not collapse.
8. Were reward weights unchanged? Yes.
9. Is scale-up still blocked? Yes.
10. What is the recommended Stage 24?
    `stage_24_policy_gradient_pilot_analysis_and_scale_readiness_review`.
