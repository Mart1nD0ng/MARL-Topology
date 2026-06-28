# Q7 — Add-only repair (D_quorum-guided, on anchor-failure scenes)

## Hypothesis (the single thing this stage tests)
On scenes where the deployable `local_hysteresis` anchor is INFEASIBLE (true C < τ), ADDING edges
(residual add-only, Q6) guided by the Q4-aligned proxy `D_quorum` can REPAIR feasibility (raise true C
≥ τ) while RETAINING every anchor edge (add-only → 100% retention by construction). If even a
D_quorum-guided add repair cannot beat the anchor on its own failure scenes, the proxy is not useful for
repair (re-check Q4 alignment / edge features, workflow Q7 fallback).

## Two groups (Contract §10.1 — never conflate)
1. **CENTRAL REFERENCE (training-only, evaluator-using):** a greedy D_quorum-guided add repair — from
   the anchor, repeatedly add the non-anchor incident edge that most reduces `D_quorum` (budget +
   mutual respected), until feasible or no candidate helps. Uses the bridge (evaluator) per candidate →
   NOT deployable. Establishes the CEILING: can D_quorum guide repair at all?
2. **DEPLOYABLE (the residual policy):** a per-edge REPAIR HEAD trained to predict
   `r_e^repair ≈ D_quorum(x) − D_quorum(x⊕e)` from LOCAL features (training targets computed via the
   bridge, training-only); at deploy it adds the top-predicted edges (add-only residual, local, 0 eval).

## Single change (one variable)
A new `src/marl_topology/training/residual_repair.py` (greedy repair + the repair-head target) + a
diagnostic `scripts/diagnostics/add_only_repair.py`. The final metric is ALWAYS the true closed-form
PBFT C; D_quorum is the Q4-authorized AUXILIARY guide only.

## What is measured (workflow Q7)
On anchor-FAILURE frames (anchor C < τ), random + urban:
- **anchor feasibility** (the deployable baseline; the failures by definition).
- **add-repair feasibility** (greedy-central and deployable-head), fraction repaired to C ≥ τ.
- **D_quorum reduction**, **C improvement** (true), **added edges** per repair.
- **RETENTION** (anchor edges kept — must be 100% in add-only; verified, not assumed).
- **mutual acceptance** (the repaired topology is mutual-decoded).
- **evaluator calls** (the central greedy's budget; the deployable head's 0 at deploy).

## Controlled variables
Anchor = `local_hysteresis` (keep 0.4 / add 0.6); τ = 0.9; add-only mode; same scenes/seeds; failure
frames only; budgets respected.

## Failing-first tests (fail on HEAD)
- `test_greedy_repair_is_add_only` — the repaired topology ⊇ the anchor (no anchor edge removed) and
  respects budgets.
- `test_greedy_repair_reduces_dquorum` — each accepted add strictly reduces D_quorum (monotone).
- `test_repair_head_target_is_delta_dquorum` — the per-edge target = `D(x) − D(x⊕e)` from the bridge.
- `test_repair_retention_is_total` — retention == 1.0 (add-only keeps every anchor edge).

## Success criterion (Q7 passes iff)
1. Tests pass; the diagnostic runs real-shard and writes a report (grouped central vs deployable).
2. The CENTRAL greedy D_quorum repair raises feasibility above the anchor on failure scenes (the proxy
   CAN guide repair) — OR an honest negative (it cannot → re-check alignment/features).
3. RETENTION == 1.0 reported; added edges + evaluator-call budget reported; final metric = true C.

## Failure criterion / fallback
D_quorum-guided add repair does NOT beat the anchor on failure scenes → honest negative; do not claim
repair works; diagnose (Q4 alignment is conditional; the failure region may need removes/swaps, i.e. Q8).

## Out of scope
Prune (Q8); PBRS (Q9); full PPO over the residual action space (Q9). Q7 establishes the add-only repair
mechanism + whether D_quorum guides it; the deployable head is a supervised regressor, not full RL.
