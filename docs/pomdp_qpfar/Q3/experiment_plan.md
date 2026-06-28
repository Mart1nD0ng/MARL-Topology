# Q3 — Quorum shortfall / D_quorum diagnostics

## Hypothesis (the single thing this stage establishes)
A quorum-deficit feasibility distance `D_quorum` can be computed from the SAME Poisson-binomial as the
true reliability `C`, and unlike `C` it has a usable gradient on the sub-feasible plateau (where many
nodes are short of quorum, `C ~ 0` and a local edit barely moves it). This is the proxy the
feasibility-plateau bottleneck (Axis B) needs. Q3 only COMPUTES and VALIDATES the primitive; it is wired
into nothing and MUST NOT enter any reward until the Q4 alignment test passes (Spec S8.3).

## Single change (one variable)
A new `src/marl_topology/protocol/quorum_deficit.py`. No production/training-path change; not imported by
any reward.

## Design (code-grounded)
Mirror `pbft_reliability.evaluate_pbft_three_phase_reliability` EXACTLY:
- prepare: receiver j needs `external_quorum` of `{α₁[i]·P_prep(i,j)}_{i≠j}` (α₁ = pre_prepare_readiness).
- commit: receiver j needs `external_quorum` of `{α₂[i]·P_commit(i,j)}_{i≠j}` (α₂ = prepared_probability).
- global: the network needs `total_quorum` of filtered `{α₃[j]}` (α₃ = committed_probability).
- SAME `remove_largest` fault filter with the SAME fault_tolerance.
Deficit math: `δ_{j,h} = E[(q_h − S_{j,h})_+] = Σ_{k<q}(q−k)P(S=k)` via an exact O(n·q) Poisson-binomial
DP whose tail bucket is identically `heterogeneous_quorum_tail` (the grounding identity).

## Controlled variables
Synthetic message matrices over small validator sets (n=4,5; f=1); the real reliability functions
(`evaluate_pbft_three_phase_reliability`, `evaluate_expected_initiator_pbft_reliability`) for grounding.

## Failing-first tests (fail on HEAD; module is new)
- `test_low_pmf_matches_bruteforce`, `test_expected_shortfall_matches_bruteforce` — vs 2^n enumeration.
- `test_tail_bucket_equals_quorum_tail` — the grounding identity (D shares C's math).
- `test_shortfall_zero_when_quorum_always_met`, `..._high_when_no_messages`, `..._decreases_when_prob_increases`.
- `test_deficit_uses_phase_receiver_structure`, `test_perfect_delivery_zero_deficit_full_reliability`.
- `test_deficit_has_gradient_where_reliability_is_flat` — the motivation (C flat at 0, D drops).
- `test_deficit_cvar_is_upper_tail_mean` (+ a float-`ceil` edge case).

## Success criterion (Q3 passes iff)
1. All tests pass; affected unit suite green.
2. `poisson_binomial_low_pmf`/`expected_quorum_shortfall` match brute force to ~1e-12.
3. The tail bucket equals `heterogeneous_quorum_tail` and the global tail reproduces
   `consensus_success_probability` (D is the reliability's exact math).
4. The plateau-gradient property is demonstrated (C flat at ~0 while D decreases).
5. `quorum_deficit` is imported by NO reward/training path.

## Failure criterion
Any divergence from brute force; tail bucket ≠ quorum tail; an off-by-one in the quorum sizes / cascade
readiness; D imported into a reward before Q4.

## Out of scope (Q4)
The real-topology ΔD-vs-ΔC alignment under local edits + the topology→message-matrices bridge +
`scripts/diagnostics/quorum_deficit_alignment.py` are Q4. Reward use is Q9, gated on Q4 PASS.
