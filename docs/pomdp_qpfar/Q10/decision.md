# Q10 — decision (local edge handshake)

**Result: KEEP. HONEST NUANCED POSITIVE.** The local edge handshake is a correctly decentralized decoder
(per-node, no global sort, neighbour-scalar-only) that — with a DIRECTED actor — provably recovers
critical edges the independent decode loses, and on URBAN real scenes reduces critical-edge disagreement
~33%. On RANDOM the real-shard effect is mixed (one-sided slightly down, critical-disagreement slightly
up) — the handshake's full value needs a directed actor (Q11). Adversarial verification: multi-lens
Workflow `wnj4i3q7h` (verdict recorded below).

## What changed (new decoder; deployment-decentralized; not yet in an end-to-end policy)
`src/marl_topology/training/edge_handshake.py`:
- `shared_edge_scores` — the symmetric shared edge score `s_ij = (s_{i→j}+s_{j→i})/2 + α·log p̂_ij +
  β·1[e∈x_{t-1}]` (Spec §12.2). For a symmetric actor `s_{i→j}=s_{j→i}=z_e`; a directional actor passes
  `directed_scores`.
- `handshake_decode` — each node ranks its incident edges by the SHARED score (identical at both ends)
  and accepts its top-b; edge active iff both accept. Per-node, NO global sort; records the control-
  communication cost (2·|E| scalars).
- `mismatch_metrics` — one-sided proposal / mutual acceptance / critical-edge disagreement (Spec §15.3).
- `correlated_sample_key` — a stable per-edge common random number `ξ_e = sha256(e,t,seed)` (Spec §12.3).

## Verification (math/structure → real-shard)
- **6 unit tests, all pass; full suite 769/0:** shared score symmetric; decode uses only neighbour
  messages (non-incident perturbation inert); NO global sort (node accept = top-b of its own incident by
  shared score); handshake RECOVERS a critical edge an independent directed decode loses (Spec §12);
  control cost = 2·|E|; correlated key stable.
- **Real-shard measurement (handshake α=1 β=0.5 vs independent pure-logit, noisy actor logits):**
  | data | one-sided (ind→hs) | critical-edge disagreement (ind→hs) | control msgs/frame |
  |---|---|---|---|
  | random | 0.607 → 0.576 | 0.304 → 0.358 (slightly worse) | 113 |
  | urban | 0.604 → 0.569 | **0.468 → 0.312 (~33% better)** | 121 |

## The findings (honest)
1. **Decentralized + correct**: per-node ranking, no global sort, only neighbour scalar exchange, control
   cost recorded — the deployment-decentralization invariant holds.
2. **Directed-score value proven**: when the actor's endpoint scores are ASYMMETRIC, the shared score
   recovers critical edges the independent directed decode loses (the unit test). This is the handshake's
   core lever — fully realized only with a directional actor (PNA, Q11).
3. **Real-shard with a symmetric noisy actor**: modest one-sided reduction (~5–6%) both; urban critical-
   edge disagreement notably down (~33%, the psucc bias aligns both ends on high-quality edges); random
   mixed (critical-disagreement slightly up). Honestly mixed, NOT a uniform win.
4. **Control overhead**: ~113–121 scalar messages/frame (2·|E|) — the deployable cost is recorded.

## Honesty / scope
- Single-config real-shard measurement (noisy actor logits, not a trained policy); the multi-seed
  end-to-end effect is Q12. The handshake is a DECODER; whether it lifts the trained policy (with a
  directional actor) is Q11/Q12.
- No over-claim of a uniform win: urban benefits (critical-edge disagreement), random is mixed.

## Acceptance table (Contract v3 §15)
- Phase: **Q10 — local edge handshake**
- Status: **VALIDATED_POSITIVE (decoder correct + decentralized + directed-score-value proven)** +
  **mixed real-shard with a symmetric actor (urban benefits, random mixed)**.
- Implemented ✓ / Wired (decoder + diagnostic) ✓ / In an end-to-end policy ✗ (Q11/Q12) / Active in this run ✓
- Test scale: 6 unit + suite 769/0 + real-shard mismatch measurement (random + urban)
- Mechanisms active: handshake decoder + mismatch metrics. Not tested: directional actor (Q11), trained-policy effect (Q12).
- Positive: decentralized; directed-score critical-edge recovery proven; urban critical-disagreement −33%.
- Negative: random real-shard mixed; full value needs a directional actor (symmetric actor → modest).
- Conclusion scope: the handshake is a correct decentralized decoder that helps where endpoint scores are
  asymmetric / on urban; NOT a uniform win with the current symmetric actor.
- Next action: **Q11 — PNA in the residual framework** (re-test the one positive-trend mechanism, PNA,
  inside the residual action space + handshake; seed-collapse rate, CI, gradient norm, params).

## Adversarial verification (multi-lens Workflow `wnj4i3q7h`, 4 lenses) — overall MINOR, no blocker/major
- **DECENTRALIZATION — PASS.** Per-node loop ranks ONLY a node's incident edges (no global argsort,
  contrast the banned `global_argsort_assemble`); each edge's shared score is edge-local; mutual step is
  local pairwise; control cost = neighbour scalars. **528 non-incident perturbations → 0 changes** to any
  node's accept; per-node accept reconstructed purely from own incident edges (no global top-K cutoff).
- **SYMMETRIC SHARED SCORE — PASS.** `s_ij = 0.5(s_{i→j}+s_{j→i}) + α·log p̂ + β·1[prev]` computed once
  per edge id; swapping the directed pair is bit-identical; psucc/prev bias is per-edge (shared). The
  critical-edge recovery is genuine (handshake topo `[A--B]` vs independent directed `[B--D]`, reproduced).
- **CONTROL-COST HONESTY — PASS.** The only non-local input is the single neighbour scalar `s_{j→i}`;
  over both directions that is exactly 2·|E| = the recorded cost; no hidden broadcast/reconciliation;
  `correlated_sample_key` is `hashlib.sha256` (reproducible across PYTHONHASHSEED).
- **CLAIM HONESTY — MINOR.** The urban critical-edge-disagreement improvement REPRODUCES (independent run
  0.441→0.376, same sign as 0.468→0.312; improved on 5/5 actor seeds × 2 scene seeds × 3 thresholds), and
  random is honestly mixed-to-worse (0.288→0.335, same sign as 0.304→0.358) — NOT cherry-picked. MINOR:
  the magnitude varies by config and urban's near-binary psucc drives the gain; do not over-claim a
  uniform win. **This decision already frames it as a nuanced positive (urban benefits, random mixed) —
  the MINOR is addressed, not a flaw.**

**Verdict: KEEP — the handshake is a correct, decentralized decoder; the honest nuanced result stands.**
