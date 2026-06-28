# Q8 — Conservative prune (D_quorum-safety-guided, on feasible anchors)

## Hypothesis (the single thing this stage tests)
On scenes where the deployable `local_hysteresis` anchor is FEASIBLE (true C ≥ τ), REMOVING low-risk
edges (residual remove-only, Q6) guided by the D_quorum SAFETY signal `risk_e = D(x∖e) − D(x)` can lower
energy/latency while KEEPING feasibility (C ≥ τ) and NEVER deleting a critical edge. If the prune drops
feasibility or can't lower cost, it is not safe — record honestly.

## The safety signal (Spec §11.2)
`risk_e = D_quorum(x∖e) − D_quorum(x)` — how much REMOVING edge e increases the quorum deficit. Low/zero
risk = e is redundant (safe to remove); high risk = e is load-bearing (keep). The DEPLOYABLE safety head
predicts `risk_e` from LOCAL features (targets training-only via the bridge); at deploy it removes the
lowest-risk edges (remove-only, local, 0 eval).

## Single change (one variable)
Add `greedy_conservative_prune` + `safety_head_targets` to `src/marl_topology/training/residual_repair.py`
+ a diagnostic `scripts/diagnostics/conservative_prune.py`. Final metrics = true closed-form PBFT
C / energy / latency; D_quorum is the Q4-authorized AUXILIARY safety guide only.

## What is measured (workflow Q8) — on anchor-FEASIBLE frames (C ≥ τ), random + urban
- **feasibility retention**: fraction of pruned topologies that stay feasible (target ~1.0 — the prune
  checks true C before each removal, so 0 feasibility loss by construction; verified not assumed).
- **energy reduction / latency reduction** (true): anchor E/L vs pruned E/L.
- **removed edges** per prune.
- **critical-edge deletion rate**: a critical edge = removing it breaks feasibility; the conservative
  greedy NEVER removes one (checks C ≥ τ first) → 0 by construction (the deployable head's rate is Q9).
- **evaluator calls** (central greedy's budget; deployable head's 0 at deploy).

## Two groups (Contract §10.1)
1. CENTRAL REFERENCE (training-only): greedy conservative prune — repeatedly remove the lowest-risk edge
   that KEEPS feasibility, until none can be safely removed. Uses the evaluator → NOT deployable. The
   ceiling: how much cost can be safely pruned?
2. DEPLOYABLE safety head (`safety_head_targets`): the per-edge `risk_e` target for a local-feature head;
   deployed prune is Q9 (the central ceiling bounds it).

## Failing-first tests (fail on HEAD)
- `test_prune_is_remove_only` — pruned ⊆ anchor (no new edge); budgets trivially respected.
- `test_prune_retains_feasibility` — on a feasible anchor, the pruned topology stays C ≥ τ.
- `test_prune_no_critical_deletion` — every removal kept feasibility (the greedy's invariant).
- `test_safety_head_target_is_risk` — `safety_head_targets[e] == D(anchor∖e) − D(anchor)`.
- `test_prune_does_not_increase_energy` — pruned energy ≤ anchor energy.

## Success criterion (Q8 passes iff)
1. Tests pass; the diagnostic runs real-shard and writes a grouped report.
2. On feasible anchors, the conservative prune lowers energy (and/or latency) while retaining 100%
   feasibility and 0 critical-edge deletions — OR an honest negative (the anchor is already minimal /
   no edge is safely removable).

## Failure criterion / fallback
The prune drops feasibility (a critical edge deleted) or cannot lower cost on any feasible anchor →
honest negative; the anchor is already cost-minimal, or D_quorum risk mis-ranks (re-check Q4 alignment).

## Out of scope
PBRS (Q9); the deployable safety head trained end-to-end (Q9); full residual policy (Q9).
