# Q8 — decision (conservative prune)

**Result: KEEP. HONEST PARTIAL POSITIVE.** The D_quorum-safety-guided conservative prune is SAFE
everywhere (100% feasibility retention, 0 critical-edge deletions) and lowers energy on URBAN feasible
anchors (47% of cases, slightly improving C via interference reduction), but gives NO cost benefit on
RANDOM (sparse anchors are already minimal; removing edges forces relay overhead → energy up). The
DEPLOYABLE safety head is deferred to Q9.

## What changed (mechanism + central reference; not in the deployed actor yet)
Added to `src/marl_topology/training/residual_repair.py`:
- `greedy_conservative_prune` — CENTRAL REFERENCE (training-only): from a FEASIBLE anchor, greedily
  REMOVE the lowest-risk edge (`risk_e = D(x∖e) − D(x)`) that KEEPS feasibility (checks true C before
  each removal → never deletes a critical edge), lowering cost where possible. Uses the evaluator → NOT
  deployable.
- `safety_head_targets` — the per-edge `risk_e` target for a deployable safety head (training-only).
- `scripts/diagnostics/conservative_prune.py` — runs it on anchor-FEASIBLE frames; final metrics = true
  C / energy / latency.

## Result (pilot: 8 scenes × 6 frames; anchor-FEASIBLE frames only; final metrics = true PBFT)
| data | feasible anchors | retention | critical deletions | mean energy reduction | frac energy down | mean removed | mean ΔC |
|---|---|---|---|---|---|---|---|
| random | 11 | **1.0** | **0** | −0.0020 (slightly UP) | 0.0 | 1.4 | 0.0 |
| urban | 32 | **1.0** | **0** | **+0.0092 (DOWN)** | **0.47** | 6.0 | +0.011 |

## The findings (honest)
1. **SAFE everywhere** — 100% feasibility retention and **0 critical-edge deletions** on both data
   sources (the greedy checks true C ≥ τ before each removal). The safety property is VALIDATED.
2. **Urban: genuine cost savings** — energy drops on 47% of feasible anchors (mean +0.0092 reduction),
   removing 6 edges, and even slightly RAISES C (+0.011) by reducing STDMA interference. Dense urban
   anchors carry prunable redundancy.
3. **Random: no benefit** — energy is flat/slightly UP and only 1.4 edges are removed. Root cause: the
   one-hop-relay PBFT regime means removing a direct edge can force a 2-hop relay (MORE transmissions →
   MORE energy), and the sparse single-RSU anchors are already near-minimal. So energy is NOT monotone
   in edge count — pruning helps only where there is true redundancy (urban).
4. The exit condition (feasibility-retained + E/L down) is MET on urban, NOT on random.

## Honesty / scope
- Single-seed pilot; the urban 47%/+0.0092 is a DIAGNOSTIC, not a multi-seed headline.
- CENTRAL REFERENCE (uses the evaluator per candidate). The DEPLOYABLE safety head (predicting `risk_e`
  from local features, 0-eval at deploy) is **deferred to Q9** (the central ceiling bounds it).
- The earlier naive intuition "removing edges lowers energy" is FALSE here (relay overhead) — recorded.
  Energy is the true closed-form PBFT energy; D_quorum is the Q4-authorized auxiliary safety guide only.

## Acceptance table (Contract v3 §15)
- Phase: **Q8 — conservative prune**
- Status: **VALIDATED_POSITIVE (safety: 100% retention / 0 critical, both data)** + **partial cost-positive
  (urban) / NEGATIVE_BUT_SCOPE_LIMITED (random; deployable head deferred)**.
- Implemented ✓ (greedy prune + safety target) / Wired into trunk ✗ (deployable head is Q9) / In this run ✓
- Test scale: 5 unit + suite 754/0; pilot single-seed 8 scenes (11 random + 32 urban feasible anchors)
- Mechanisms active: D_quorum-safety conservative prune (central reference). Not tested: deployable head (Q9).
- Positive: safe (100% retention, 0 critical) everywhere; urban energy down on 47% + slight C gain.
- Negative: no cost benefit on random (relay overhead; sparse anchors minimal); central, not deployable.
- Conclusion scope: conservative pruning is SAFE and lowers cost where redundancy exists (dense urban),
  but not on sparse random anchors (relay overhead). Central-reference ceiling — NOT a deployable headline.
- Next action: **Q9 — full residual + PBRS**: combine add/remove/swap into the deployable residual policy
  trained end-to-end with potential-based shaping `F_t = γΦ(s_{t+1}) − Φ(s_t)`, `Φ = −D_quorum`, terminal
  Φ = 0; tests: PBRS telescopes, does not change the exact small-MDP optimum, final metrics use true C/E/L.
