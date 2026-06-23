# Environment-Math Reconstruction + Recalibration Report

**Scope:** the spec-driven reconstruction loop of 2026-06-22/23 — rebuilding the
`MARL-Topology` environment math (P0–P4) to the two authority docs
(`docs/MARL-Topology-Technical-Spec.md`, `docs/MARL-Topology-Engineering-Plan.md`),
then **recalibrating** the production pipeline under the corrected math and re-establishing
the headline honestly. 19 commits, **zero new test failures throughout**, one evidence-driven
REVISE. Full per-iteration detail: `docs/URBAN_V2X_RESEARCH_LOG.md`; live status:
`docs/CURRENT_HEAD_STATUS.md`.

---

## 1. Executive summary

The project's environment math had several latent correctness bugs (unsafe PBFT quorum at
the scales it actually runs, an incoherent per-phase fault model, a double-counted relay, a
degenerate latency, a search-miss-treated-as-infeasible label). All were fixed as verified,
opt-in primitives, then **activated** in the production regime and the dataset rebuilt.

**Key finding — the corrected math is feasibility-neutral.** Re-evaluating the production
scenarios under the corrected math leaves the feasibility distribution *identical* (0.667,
same family bins), so no scenario re-calibration was needed.

**Headline (honest, under correct math, in-range N∈{8,12,16}, 5 seeds):** the oracle-free
cold-start **decentralized learner is statistically indistinguishable from the centralized
Simulated-Annealing oracle** — RL keep-best 0.624 vs SA ceiling 0.655, **margin −0.031, 95%
CI [−0.086, +0.024]** (final-update policy +0.007, [−0.021, +0.036]); both CIs span 0. It
**matches** the near-optimal oracle in-range, neither beats nor trails.

**Retired:** the earlier "+0.177 beat at out-of-range N=24" was produced under the
*pre-recalibration (incorrect)* math and is `retired_due_to_protocol_metric_change`. Whether a
learned policy beats the oracle **out-of-range (N=24)** under correct math is the open question.

---

## 2. The reconstruction (P0–P4)

| Phase | Bug | Fix | Commit |
|---|---|---|---|
| 0 | Docs conflated CTDE with critic-free; suite 287-contract-red (stale lineage) | Froze HEAD status; adopted CTDE governance (D1–D6); demoted "critic-free" to a baseline | `1e30f89` |
| 1a | PBFT quorum hardcoded `q=2f+1` (only safe at n=3f+1); at n>3f+1 the intersection `2q−n` can be ≤ f → two quorums with no honest overlap (safety violation) | `PBFTQuorumSpec` (classic_exact / safe_generalized, `q=⌊(n+f)/2⌋+1`); asserts `2q−n>f`, `q≤n−f`, `n≥3f+1` | `abefe88` |
| 1b | `remove_largest` strips the f best senders *independently per receiver and per phase* — not a single coherent adversary (Spec §4.7) | `C_robust = min_{|B|≤f} C(x;B)` with one fixed Byzantine set across all phases; exact + softmin + greedy | `36d927d` |
| 1b-wire | (REVISE) wiring caught a τ-cap bug: zeroing faulty primaries caps `C ≤ (n−f)/n` < τ → τ unreachable | Fixed: average over *honest* initiators only (deferred view-change); wiring deferred | `c292faa` |
| 1c | Quorum tail only existed as a non-differentiable reference DP | Torch differentiable `torch_quorum_tail` — reference parity <1e-12, gradcheck, analytic Spec-§4.5 sensitivity | `630a890` |
| 2 | Route/relay double-counted multi-hop (BFS-route delivery fed into a second relay DP); A–B–C at relay_hops=1 wrongly gives P(A→C)>0 | `one_hop_relay`: build the matrix from direct links only so the relay DP is the single multi-hop layer | `72ffe05` |
| 3 | Binary `feasible_exists` conflates a finite-search miss with infeasibility (#11) | `solvability/` tri-state (witness_feasible / certified_infeasible / unknown); a finite-search miss is `unknown`; split-isolated witness memory | `ec43f59` |
| 4a | Latency `min(max_all_pairs, budget)` — degenerate, topology-insensitive, a failed topology paid ~0 | Quorum-completion timeout-aware latency `E[min(T,B)]`; a failed topology pays the full budget | `c68ac04` |

Every fix landed verified + opt-in + inert (default-off byte-identical), so the suite stayed
green throughout. The frozen banned-literal `src/**` gates (`MAPPO`/`COMA`/`Critic(`…) and the
287 stale stage-contract tests remain as the gating dependency for the CTDE model phases (7–9).

---

## 3. The recalibration (activation + rebuild)

| Step | Action | Commit |
|---|---|---|
| 1 | Configurable knobs (`fault_model`, `one_hop_relay`) on the evaluator config; **measured** corrected ≈ baseline feasibility at relay_hops≥2 | `1f7ebc1` |
| 2a | Flipped the **production** regime (`operating_point_regime`) to `fixed_set` + `one_hop_relay` (relay_hops=3); **measured identical feasibility distribution (0.667)** → no τ-gradient re-tuning needed | `d65a71d` |
| 2b | Wired the timeout-aware latency into both evaluators (config-gated, on in production) | `dcc7ffe` |
| 2c | Recorded the tri-state `solvability_status` on scenario specs (finite-search miss = unknown) | `29a9f2a` |
| pilot | Built a corrected N=8 shard (0.69 feasible) + cold-start train → feasibility 0.03→1.0, held RL 0.571 → **pipeline validated end-to-end** | `b3f733b` |
| 2e | Built a 144-scene corrected dataset (4 shards N∈{8,12,16}) + 5-seed cold-start headline | `3c60612`, `8d6b24a` |

The biggest feared risk — a feasibility collapse needing heavy re-tuning — did not materialize.
The Phase-1b-wire scare was fully explained as the τ-cap bug (fixed) + relay_hops=1, not the
fixed-B model. Cost: the corrected `fixed_set` is O(n⁴) → ~2× per scene in-range (manageable);
the out-of-range N=24 build is the heavy tail (~81× N=8), gated on a cost optimization.

---

## 4. The in-range corrected headline (5 seeds, 144 scenes, held 58/seed)

| metric | mean | 95% CI (n=5) |
|---|---|---|
| RL keep-best (deployed) | 0.624 | [0.569, 0.679] |
| SA ceiling (oracle) | 0.655 | [0.598, 0.712] |
| **Margin keep-best (RL − ceiling)** | **−0.031** | **[−0.086, +0.024]** |
| Margin RLfin (final-update policy) | +0.007 | [−0.021, +0.036] |

Per-seed (RL keep-best / ceiling): 0.586/0.603, 0.655/0.638, 0.638/0.724, 0.569/0.638,
0.672/0.672. Conditional reliability 0.88–0.97 (the deployed policy reaches τ on the solvable
held scenes). keep-best is selected on a held-out-from-train VAL split (never on the held set).

**Interpretation:** in-range, the SA oracle is a near-optimal centralized search, and the
decentralized learner matches it — there is no in-range gap to "beat." A "beats the oracle"
result is only meaningful **out-of-range**, where a learned policy can generalize past the
oracle's build scale; that comparison under correct math is unresolved.

---

## 5. Verified vs open

**Verified:** safe quorum + fixed-B fault model + one-hop relay + Torch quorum-tail + tri-state
solvability + timeout latency, all unit-tested; the corrected pipeline builds healthy data and
trains a feasible policy; in-range parity with the oracle (5-seed CI).

**Open / next:**
1. **Out-of-range N=24 headline** — needs a cheaper exact-`f=1` `fixed_set` (avoid the O(n⁴)
   re-cascade) + a heavy N=24 build, to test the legacy claim under correct math.
2. **CTDE model phases (7–9)** — Graph-MAPPO → Counterfactual PPO → SCQ (the spec's target
   architecture) — blocked on retiring the 287 stale stage-contract tests + the banned-literal
   `src/**` gates.

---

*Generated by the autonomous reconstruction loop. Per-iteration detail:
`docs/URBAN_V2X_RESEARCH_LOG.md`. Authority: `docs/MARL-Topology-Technical-Spec.md`,
`docs/MARL-Topology-Engineering-Plan.md`.*
