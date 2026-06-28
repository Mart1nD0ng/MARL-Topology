# Q3 — decision (quorum shortfall / D_quorum diagnostics)

**Result: KEEP.** A new `protocol/quorum_deficit.py` computes the quorum-shortfall feasibility distance
`D_quorum` (Spec S8) — a TRAINING-ONLY proxy that is **NOT yet wired into any reward** (gated on the Q4
alignment test, per Spec S8.3). It is the SAME Poisson-binomial math as the reliability (proven, not a
toy), and it demonstrably has a gradient where the true `C` is flat. Adversarially verified (independent
agent, 6 claims, **PASS**; 1 MINOR float-edge fixed).

## What changed (new module, wired into nothing)
`src/marl_topology/protocol/quorum_deficit.py`:
- `poisson_binomial_low_pmf(probs, q)` → `[P(S=0..q-1), P(S>=q)]`, exact O(n·q) DP. The tail bucket `[q]`
  is **identically** `heterogeneous_quorum_tail(probs, q)` (pinned by a test) → D shares C's exact math.
- `expected_quorum_shortfall(probs, q) = Σ_{k<q}(q-k)P(S=k) = E[(q-S)_+]` (brute-force validated).
- `per_primary_quorum_deficit(...)` — mirrors `evaluate_pbft_three_phase_reliability`'s cascade EXACTLY:
  prepare receiver-quorum on `{α₁[i]·P_prep(i,j)}` (external_quorum), commit on `{α₂[i]·P_commit(i,j)}`
  (external_quorum), global on filtered `{α₃[j]}` (total_quorum); SAME `remove_largest` fault filter.
- `expected_initiator_quorum_deficit(...)` — uniform-initiator average (mirrors the expected-initiator C).
- `deficit_cvar` / `aggregate_deficits` → D_mean / D_max / D_cvar + worst_phase / worst_receiver.

## The motivation, demonstrated (the whole point of D_quorum)
On a 5-validator synthetic channel, sweeping per-link delivery p:

| p | true C | D_quorum_mean | D_global |
|---|---|---|---|
| 0.50 | 0.000 | 2.745 | 4.000 |
| 0.70 | 0.000 | 2.344 | 4.000 |
| 0.90 | ~0 (1e-6) | 1.385 | 3.847 |
| 0.99 | 0.288 | 0.230 | 1.071 |
| 0.999 | 0.883 | 0.025 | 0.122 |

**From p=0.5 to p=0.9 the true C stays flat at ~0 (the sub-feasible plateau), but D_quorum drops
2.74 → 1.39** — exactly the gradient the feasibility-plateau bottleneck needs. C is a whole-network
product that underflows; D is a smooth `E[(q-S)_+]` expectation that still moves.

## Verification (math truth → grounding → property)
- **10 unit tests, all pass; full suite 730/0** (720 → 730).
- Brute-force: `poisson_binomial_low_pmf` and `expected_quorum_shortfall` match 2^n enumeration to ~1e-12.
- Grounding identity: tail bucket == `heterogeneous_quorum_tail`; the global tail bucket reproduces
  `consensus_success_probability` to ~1e-19 (the deficit IS the reliability's math).
- Properties: zero when quorum always met; = q when no messages; strictly decreasing in message prob;
  CVaR = upper-tail mean (worst (1-α) fraction).

## Adversarial verification (independent agent, 6 claims) — PASS
1. SAME MATH ✓ (tail bucket = quorum tail, 300 random vectors, worst err 3.3e-16).
2. CORRECT SHORTFALL ✓ (400 brute-force cases, worst err 2.7e-15; q=0/q>n/all-ones/all-zeros exact).
3. CASCADE MIRRORS RELIABILITY ✓ (α₁/prepare, α₂/commit, filtered-α₃/global, same fault filter; global
   tail reproduces per-primary consensus to 4.3e-19; no off-by-one).
4. NOT WIRED INTO REWARD ✓ (only `tests/` imports it; nothing in training/ or scripts/train).
5. NUMERICAL SOUNDNESS ✓ (valid sub-distribution, fsum; CVaR is upper tail). MINOR: float `ceil` edge in
   `deficit_cvar` ((1-0.7)·10 = 3.0000…04 → 4) — **fixed** with `round(...,9)`, pinned by a new test case.
6. HONEST CLAIM ✓ (the plateau-gradient is real: C underflows to machine-zero while D decreases smoothly).

## Scope / honesty
- D_quorum is **NOT in any reward** and MUST NOT be until Q4's alignment test passes (Spec S8.3).
- Validated on the math + the real reliability cascade with declared matrices. The **real-topology**
  ΔD-vs-ΔC alignment under local edits (Spec S8.3) is **Q4's** explicit deliverable
  (`scripts/diagnostics/quorum_deficit_alignment.py`) — that is where D is computed on sampled real
  topologies and checked for sign-alignment with the true C, plus the topology→matrices bridge.
- No `mechanism_activation` artifact: Q3 wires into no training run (a primitive, not an active mechanism).

## Acceptance table (Contract v3 §15)
- Phase: **Q3 — quorum shortfall / D_quorum diagnostics**
- Status: **VALIDATED_POSITIVE (primitive correct + grounded)** — math primitive, not in reward.
- Implemented ✓ / Wired into entrypoint ✗ (intentionally — gated on Q4) / Active by default ✗ /
  Active in this run n/a (not a runtime mechanism yet)
- Test scale: 10 unit (brute-force + reliability grounding); full suite 730/0
- Mechanisms active: none in training. Not tested: real-topology ΔD-vs-ΔC alignment (Q4); reward use (Q9).
- Positive findings: D_quorum is the reliability's exact Poisson-binomial; has gradient where C is flat.
- Negative findings: none in scope. Open risk for Q4: whether ΔD anti-correlates with ΔC under LOCAL
  edits on the REAL topology distribution (the synthetic monotone preview is encouraging, not sufficient).
- Conclusion scope: the D_quorum primitive is correct and reliability-grounded; NOT yet a reward signal.
- Next action: **Q4 — D_quorum ↔ true C alignment test** (Spearman ΔD/ΔC, top-k repair hit, ΔD↑-C↓ rate
  under local edits on real `--dyn-data` topologies). D_quorum may enter reward (Q9) ONLY if Q4 passes.
