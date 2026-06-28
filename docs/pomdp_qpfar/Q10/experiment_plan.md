# Q10 — Local edge handshake (shared edge score, symmetric mutual selection)

## Hypothesis (the single thing this stage tests)
A two-round local handshake — each node sends its endpoint score `s_{i→j}` to neighbour j, both compute
the SAME shared edge score `s_ij = (s_{i→j}+s_{j→i})/2 + α·log p̂_ij + β·1[e∈x_{t-1}]`, and both rank
their budget-incident edges by `s_ij` — reduces the mutual-acceptance MISMATCH (one-sided proposals /
critical-edge disagreement) versus the independent per-edge decode, WITHOUT a global sort and using only
neighbour scalar exchange (deployment-decentralized). The shared psucc/prev bias aligns both ends on the
same high-quality edges.

## Single change (one variable)
A new `src/marl_topology/training/edge_handshake.py` (the handshake decoder + the shared-score helper +
control-communication cost) + a measurement diagnostic. The deployed mutual-acceptance semantics are
preserved (edge active iff both endpoints accept); only the per-node ranking SCORE changes.

## Design (Spec §12; mirrors local_mutual_assemble)
- `shared_edge_scores(directed_scores, edge_ids, observed_psucc, prev, α, β)` → `{eid: s_ij}` with
  `s_ij = (s_{i→j}+s_{j→i})/2 + α·log p̂_ij + β·1[e∈prev]`. For a symmetric actor `s_{i→j}=s_{j→i}=z_e`.
- `handshake_decode(...)` → `(accept, topology, control_messages)`: each node ranks its incident edges by
  the SHARED score (identical at both ends) and accepts its top-b; edge active iff both accept. The score
  being symmetric, the only residual mismatch is budget competition. Per-node computable; NO global sort.
- Control-communication cost = the scalar messages exchanged (one `s_{i→j}` per directed edge = 2·|E|).
- Optional correlated sampling: a stable per-edge hash key `ξ_e = sha256(e,t,seed)` both ends share.

## What is measured (Spec §15.3) — vs the independent per-edge decode (α=0 baseline)
- one-sided proposal rate (an edge proposed by one end but not the other).
- mutual acceptance rate.
- critical-edge disagreement (a high-psucc/bridge edge lost to a one-sided proposal).
- shared-score agreement; control message cost.

## Failing-first tests (fail on HEAD)
- `test_shared_edge_score_symmetric` — `s_ij` is identical from both endpoints' computation (symmetric).
- `test_handshake_uses_only_neighbor_messages` — a node's accept depends only on its incident edges' shared
  scores (changing a non-incident edge's directed score leaves the node's accept unchanged).
- `test_no_global_sort` — the decode ranks per-node incident sets only (no global argsort over all edges).
- `test_mutual_acceptance_improves_on_psucc_bias` — adding the psucc bias (α>0) reduces the one-sided
  proposal rate vs the pure-logit decode (α=0) on a constructed case.
- `test_control_message_cost_recorded` — the control cost = 2·|E| scalar messages, reported.

## Success criterion (Q10 passes iff)
1. Tests pass; the decoder is decentralized (per-node, no global sort) and control cost recorded.
2. A real-shard measurement reports the one-sided / mismatch metrics for the handshake vs the independent
   decode (random + urban) — honestly whether the handshake reduces mismatch.

## Failure criterion
The handshake needs a global sort / global state (not decentralized); the shared score is asymmetric; the
handshake does not reduce mismatch (honest negative — the independent decode already aligns).

## Out of scope
PNA / directional actor in the residual frame (Q11); the multi-seed campaign (Q12). The handshake is a
DECODER improvement; whether it lifts the end-to-end policy is Q11/Q12.
