# Q6 — Residual action space (anchor ⊕ residual)

## Hypothesis (the structural property this stage establishes)
A decentralized residual action space exists where the deployable `local_hysteresis` anchor is the
DIRECTLY-COMPUTED base and the actor's per-edge residual logits only ADD / REMOVE / SWAP incident edges:
`final = anchor ⊕ residual`, decoded LOCALLY + MUTUALLY, with `zero residual = anchor EXACTLY`, the
radio budget always respected, and 0 action-evaluator calls. This is the action space the residual RL
arms (Q7 add-only repair, Q8 prune, Q9 full+PBRS) act in. It sidesteps the Q5 BC-imitation failure: the
anchor is reproduced exactly by construction, not learned.

## Single change (one variable)
A new `src/marl_topology/training/residual_action.py::residual_decode` (+ a deployable wrapper). No
reward, no RL sampling/logp yet (Q7), no PBRS (Q9). Default path unchanged (this is a new decoder, not
wired into the trunk yet).

## Design (mirrors `policies/decentralized_mutual_acceptance.local_mutual_assemble`)
`residual_decode(anchor_topology, residual_logits, edge_ids, context, *, mode, add_threshold=0.0,
remove_threshold=0.0) -> (accept, topology)`:
- Each node's BASE proposal = the anchor edges incident to it (≤ its budget b, since the anchor came
  from the budget-capped hysteresis proposals).
- REMOVE (modes remove/swap/full): drop incident ANCHOR edges with residual logit < remove_threshold.
- ADD (modes add/full): include incident NON-anchor edges with logit > add_threshold, up to the budget
  room AFTER keeping all anchor edges (so the budget cap never evicts an anchor edge in add/full).
- SWAP: remove K anchor edges, add at most K non-anchor edges (|final| ≤ |anchor| → budget preserved).
- Mutual: an edge is active iff BOTH endpoints keep it. Per-node computable; 0 evaluator calls.
- `zero residual` (no logit crosses a threshold; thresholds at 0 with strict </>): final == anchor exactly.

## Controlled variables
Anchor = `local_hysteresis_action(..., keep_threshold=0.4, add_threshold=0.6)` (the D13 deployable
baseline); residual logits are synthetic in the unit tests (the actor's residual head is wired in Q7).

## Failing-first tests (fail on HEAD; the module is new)
- `test_residual_zero_equals_anchor` — `residual_decode(anchor, zeros)[1] == anchor` (all modes).
- `test_add_only_never_removes_anchor_edges` — add mode: final ⊇ anchor (no anchor edge dropped, incl.
  under the budget cap).
- `test_remove_only_never_adds_edges` — remove mode: final ⊆ anchor (no new edge).
- `test_swap_preserves_budget` — swap: every node's final degree ≤ its anchor degree ≤ budget.
- `test_residual_decoder_is_local` — changing a logit for an edge NOT incident to node w does not change
  w's accept (locality / decentralization).
- `test_residual_respects_budget_all_modes` — every node's final degree ≤ its budget in every mode.

## Success criterion (Q6 passes iff)
1. All structural tests pass; affected suite green.
2. A real-shard smoke decodes residuals on real `--dyn-data` topologies (random + urban) with 0
   evaluator calls and zero-residual reproducing the anchor exactly on every frame.
3. Budget respected and locality holds on real scenes (not just synthetic).

## Failure criterion
zero residual ≠ anchor; add-only removes an anchor edge (budget-cap eviction); a non-incident logit
changes a node's accept (non-local); any node exceeds its budget.

## Out of scope
RL sampling + per-edge residual log-prob for PPO (Q7); D_quorum-guided repair (Q7); prune (Q8); PBRS (Q9).
The actor's residual HEAD + trunk `--residual` flag are wired in Q7 (the first residual RL arm).
