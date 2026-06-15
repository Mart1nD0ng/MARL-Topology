# V5 Routes Not To Learn Blindly

This list turns v5 experience into guardrails. It is not a rejection of the v5 research effort; it is a boundary against importing high-entropy decisions before the new clean skeleton needs them.

## Old Reward

Do not copy old reward formulas or weights.

Why:

- v5 mixed consensus shaping, latency, energy, timeout, quorum, edge terms, and later constrained variants.
- Phase38 found no reward contract passed all hard gates.
- Phase38.1 showed old critical/harmful labels were too coarse for reward gates.

Allowed learning:

- Reliability should be a constraint.
- Latency and energy should be optimized after reliability is feasible.
- Above-threshold reliability should plateau unless a registered safety-margin metric exists.

## Old Metric Names

Do not import `P_succ`, `P_eff`, `P_eff_new`, hard/soft/legacy modes, log/geom/arith summaries, or CSV names by default.

Why:

- Phase35 had to separate base consensus success from effective success.
- Legacy mode explicitly aliased effective success to base success.
- Later hard/soft modes carried deadline and quorum semantics.

Allowed learning:

- If a derived metric becomes necessary, register it with definition, range/unit, level, used_for, formula source, dependencies, and tests.

## Old Phase Script Structure

Do not copy phase scripts as the new mainline.

Why:

- Read-only inventory found 28 phase scripts and 37 phase tests.
- Durable behavior spread across `scripts/phase*`, result directories, and docs.

Allowed learning:

- Use phase reports as evidence.
- Convert durable behavior into `src/marl_topology/` modules only after contract tests exist.

## Full-Mask Optimal Assumption

Do not assume full-mask is a reliability upper bound or resource optimum.

Why:

- Phase32 classified one state set as objective full-mask optimal, but Phase36 found strict physical Pareto dominance over full in many cases and many constrained resource-redundant full-mask cases.
- Full-mask can be strong, redundant, or physically suboptimal depending on interference and resource constraints.

Allowed learning:

- Keep full-mask as a required baseline in oracle reports.

## Fixed 0.5 Deployment Threshold

Do not treat fixed 0.5 thresholding as the default deployment policy.

Why:

- v5 actor code used `>= 0.5` deterministic thresholds.
- v5 saw both full-mask reproduction and empty-graph collapse under deterministic evaluation.
- Phase39 recommended calibrated deployment rather than fixed thresholding.

Allowed learning:

- Fixed 0.5 is a baseline to test, not a deployment rule.

## Actor Global-Information Leakage

Do not assume v5 actor inputs are Dec-POMDP-safe.

Why:

- v5 actor/training interfaces carried full matrices such as `edge_feat`, `phys_adj`, `topo_prev`, and topology history.
- This may be valid for a centralized graph policy experiment, but it is not automatically valid for deployment actors under a Dec-POMDP contract.

Allowed learning:

- Edge memory may be useful if derived from local history and schema-governed.
- Centralized state may be critic-only during training.

## Unregistered Metric / Reward Coupling

Do not let metric fields become reward terms without registration and review.

Why:

- v5 reward contracts read `P_eff_new`, `P_succ_base`, latency, energy, edge terms, deadline, and quorum components.
- Later audits had to separate diagnostics, hard gates, margin cases, and resource tradeoffs.

Allowed learning:

- Every reward component must map to registered metrics or explicitly diagnostic fields.

## Q/COMA Architecture

Do not inherit v5 COMA or Q critic as default.

Why:

- Phase10 RCA found an action-summary Q critic could be action-blind.
- Phase32 and Phase37.1 showed sign, ranking, calibration, and rare-safety metrics could disagree.
- Phase39 kept COMA gated until direct edge-delta fidelity passed.

Allowed learning:

- Credit assignment needs oracle-labeled fidelity and calibration gates.

