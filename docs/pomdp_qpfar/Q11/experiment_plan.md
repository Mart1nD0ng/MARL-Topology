# Q11 — PNA in the residual framework

## Hypothesis (the single thing this stage tests)
The one positive-trend mechanism — the directional PNA actor (D12/D13: +0.125 held-feasibility but
BIMODAL, a seed-0 collapse) — when placed INSIDE the residual action space (Q6) with the anchor
trust-region (Q9 PART 2) and PBRS, is it (a) more STABLE (no bimodal seed collapse) than its from-scratch
D13 form, and (b) does it BEAT the MLP residual actor? The contract: do NOT write the PNA trend as a
significant win; decide from ≥5 seeds with CI.

## Single change (one variable)
`--actor {mlp, pna}` in the Q9 residual+PBRS trainer (`scripts/diagnostics/residual_pbrs_train.py`) — the
PNA actor (DynamicPNAActor: directional message passing + PNA aggregation + ω-preference, a
signature-compatible drop-in) produces the residual logits instead of the MLP. Everything else (residual
action space, anchor trust-region, PBRS, eval-no-shaping) is held fixed.

## What is measured (workflow Q11) — 5 seeds × {mlp, pna}, recommended config (--pbrs --anchor-reg 0.5)
- **seed-collapse rate**: fraction of seeds where the residual policy BREAKS the anchor (retention low /
  feasibility far below the anchor) — the D13 PNA failure mode.
- mean held feasibility (residual vs anchor) + CI across seeds (paired).
- **gradient norm** (mean/max over updates) — PNA's `logit_scale=10` could make residual flips more
  aggressive / gradients larger (an instability signal).
- runtime, parameter count (PNA vs MLP).

## Controlled variables
Data random + urban; residual full mode; anchor trust-region 0.5; PBRS on; eval NO shaping (true C/E/L);
same scenes/seeds across arms; current CSI (one variable = the actor architecture).

## Failing-first tests (fail on HEAD)
- `test_pna_actor_drops_into_residual_trainer` — `make_actor("pna", ...)` builds; the residual rollout
  produces per-frame differentiable log-probs (PPO-ready). [DONE]
- the reinforce_update now reports the gradient norm (finite). [DONE]

## Success criterion (Q11 passes iff)
1. Tests pass; the PNA residual trains end-to-end (no NaN) across the 5 seeds.
2. A multi-seed report (per-seed + CI) honestly states whether PNA beats MLP in the residual frame and
   its seed-collapse rate vs MLP — with NO over-claim of the PNA trend.

## Expected / honest framing
Given Q9 PART 2 (the residual policy with the trust-region converges to the anchor) and the central
ceilings, the likely outcome is PNA ≈ MLP ≈ anchor (both match the anchor; the trust-region prevents the
D13 collapse). The interesting question is whether PNA is MORE prone to collapse (logit_scale=10) than MLP
even with the trust-region. Either way the result is reported as-is.

## Adversarial verification (Ultracode)
A multi-lens Workflow: (1) the PNA/MLP comparison is fair (matched config, paired seeds, same budget);
(2) the seed-collapse + CI are computed correctly from the raw per-seed JSON; (3) no over-claim of the PNA
trend; (4) the PNA actor is genuinely active (params differ, gradients reach it).

## Out of scope
The full multi-arm campaign (Q12); the docs close-out (Q13).
