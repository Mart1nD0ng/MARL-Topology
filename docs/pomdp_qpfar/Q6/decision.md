# Q6 — decision (residual action space)

**Result: KEEP.** A decentralized residual action space — the deployable `local_hysteresis` anchor is the
DIRECTLY-COMPUTED base, the actor's per-edge residual logits ADD/REMOVE/SWAP incident edges, decoded
locally + mutually, `zero residual = anchor EXACTLY`, budget always respected, 0 evaluator calls. This is
the action space the residual RL arms (Q7–Q9) act in, and it reproduces the anchor by construction (no
BC needed — resolving the Q5 urban imitation failure). Adversarially verified (independent agent, 6
claims, **PASS, zero issues**).

## What changed (new decoder, not yet wired into the trunk)
`src/marl_topology/training/residual_action.py`:
- `residual_decode(anchor_topology, residual_logits, edge_ids, context, *, mode, add_threshold=0.0,
  remove_threshold=0.0) -> (accept, topology)` — mirrors `local_mutual_assemble` (per-node rank, top-b,
  mutual) but pivots around the anchor. Each node starts from its anchor-incident edges; REMOVE drops
  anchor edges with logit < remove_threshold (remove/swap/full); ADD includes non-anchor incident edges
  with logit > add_threshold up to the budget room AFTER keeping all anchor edges (add/full); SWAP adds
  at most as many as removed. Strict thresholds at 0 → a zero logit is inert → exact anchor.
- `deployable_residual_topology(obs, residual_logits, prev, *, mode, ...)` — convenience: compute the
  anchor (local_hysteresis, 0 eval) then the residual; the whole path is local + mutual.

## Verification (math → real-shard → adversarial)
- **6 unit tests, all pass; full suite 745/0** (739 → 745): zero=anchor (all modes), add-only never
  removes anchor edges, remove-only never adds, swap preserves budget, decode is local, budget respected.
- **Real-shard smoke (random + urban, 18 frames each):** zero residual == anchor on EVERY frame; budget
  respected in all 4 modes; 0 evaluator calls.
- **Adversarial verify (independent agent, 6 claims) — PASS, no issues:**
  1. zero-residual == anchor EXACTLY (strict thresholds make a 0 logit inert; holds for torch tensors via float()).
  2. DECENTRALIZED — per-node loop over incident edges only, per-node budget, NO global argsort
     (contrast the banned `global_argsort_assemble`); 50 non-incident perturbations × 4 scenes never
     changed an unrelated node's accept.
  3. ADD-ONLY preserves anchor edges under the budget cap (room computed AFTER keeping the anchor;
     adversarial −9/+9 logits: all anchor edges survived, incl. budget-saturated nodes).
  4. BUDGET respected in 1600+ real-scene decodes (4 modes × 200 logit vectors × 4 scenes).
  5. TORCH-FREE / 0 evaluator calls (only node_budgets + graph.edges reads; the deployed wrapper is
     evaluator-free end-to-end).
  6. MODE semantics: remove-only ⊆ anchor; swap never grows a node's proposal beyond its anchor degree.

## Honesty / scope
- Q6 is the ACTION SPACE + DECODER only — no RL training, no reward, no PBRS. The actor's residual HEAD,
  the stochastic residual sampler + per-edge log-prob for PPO, and the trunk `--residual` flag are Q7
  (the first residual RL arm: add-only repair). No headline / performance claim here.
- `residual_decode` lives under `training/` (gate-exempt) but the LOGIC is deployment-decentralized
  (verified by the locality test + the adversarial agent) — file location ≠ deployment status; the
  invariant is satisfied by the local+mutual+0-eval logic.

## Acceptance table (Contract v3 §15)
- Phase: **Q6 — residual action space**
- Status: **VALIDATED_POSITIVE (decoder correct + decentralized)** — structural stage, not a result.
- Implemented ✓ / Wired into entrypoint ✗ (Q7 wires the residual head + trunk flag) / Active in this run ✓
  (unit + real-shard smoke)
- Test scale: 6 unit + suite 745/0 + real-shard smoke (random + urban, 18 frames); 1600+ adversarial decodes
- Mechanisms active: residual decode (eval-only). Not tested: residual RL training (Q7+).
- Positive: zero-residual=anchor exact; decentralized (local + mutual, no global state); budget-safe; 0-eval.
- Negative: none in scope.
- Conclusion scope: a correct, decentralized residual action space exists and reproduces the anchor by
  construction. NOT a claim that residual RL improves anything (Q7+).
- Next action: **Q7 — add-only repair**: wire the actor's residual head (add-only mode) + the stochastic
  residual sampler + per-edge log-prob; train from anchor-FAILURE scenes, guided by D_quorum; report
  anchor feasibility, add-repair feasibility, D_quorum reduction, C improvement, added edges, RETENTION.
