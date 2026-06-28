# Q11 — decision (PNA in the residual framework)

**Result: KEEP (PNA stays opt-in). HONEST NEGATIVE — decisive.** Inside the residual+PBRS frame with the
anchor trust-region, PNA and MLP both converge EXACTLY to the anchor (paired residual−anchor diff
**0.000, CI [0,0]**, retention **1.0**, **0/5 seed collapse**, 0 diverged on 5 seeds × {random, urban}).
**PNA == MLP == anchor — the D12/D13 PNA +0.125 trend does NOT reproduce** in the controlled residual
frame (it was bimodal noise). PNA's only distinguishing traits are a much larger/spikier gradient
profile (`logit_scale=10`; raw norm up to ~28857 vs MLP's global max ~3055 / random-only ~255) and 2×
the parameters. The binding limit remains feasibility-region learning.
Adversarial verification: multi-lens Workflow `wslezjbo1` (verdict below).

## What changed (one variable: the residual actor architecture)
`scripts/diagnostics/residual_pbrs_train.py` — `make_actor("pna"|"mlp")` (the PNA actor `DynamicPNAActor`
— directional message passing + PNA aggregation + ω-preference — is a signature-compatible drop-in) +
gradient-norm and parameter-count tracking. Everything else (residual action space, anchor trust-region
0.5, PBRS, eval-no-shaping) held fixed.

## Result (5 seeds × {mlp, pna}, --pbrs --anchor-reg 0.5; eval NO shaping; final metrics true PBFT C/E/L)
| data | arch | params | diverged | seed-collapse | residual feasibility (per seed) | anchor feasibility | retention | grad-norm max (per seed) | paired (resid−anchor) |
|---|---|---|---|---|---|---|---|---|---|
| random | mlp | 43,265 | 0 | 0/5 | 0.23/0.27/0.25/0.58/0.46 | = residual | 1.0×5 | 28/48/256/35/59 | **0.000 [0,0]** |
| random | pna | 88,513 | 0 | 0/5 | 0.23/0.27/0.25/0.58/0.46 | = residual | 1.0×5 | 25/**4822**/55/**28857**/569 | **0.000 [0,0]** |
| urban | mlp | 43,265 | 0 | 0/5 | 0.79/0.79/0.81/0.88/0.77 | = residual | 1.0×5 | 44/25/27/3055/25 | **0.000 [0,0]** |
| urban | pna | 88,513 | 0 | 0/5 | 0.79/0.79/0.81/0.88/0.77 | = residual | 1.0×5 | 22/**2863**/39/1031/35 | **0.000 [0,0]** |

## The findings (honest)
1. **PNA gives NO advantage in the residual frame** — PNA and MLP produce the IDENTICAL residual policy
   (both converge to the anchor; paired diff exactly 0.000 on every seed). The D13 PNA +0.125 was a
   from-scratch, bimodal artifact; in the controlled residual frame (anchor base + trust-region) it does
   not reproduce. Per the contract, this is NOT a PNA win — it is a precise null.
2. **No seed collapse for either** (0/5) — the residual frame + anchor trust-region + gradient clipping
   prevent the D13-style bimodal collapse. PNA, which collapsed in D13, is STABILIZED here (but only
   because it is clamped to the anchor — not because it learned anything better).
3. **PNA is gradient-explosive** — `logit_scale=10` drives raw gradient norms up to ~28857 (PNA median
   max ~312) vs MLP's global max ~3055 (random-only ~255; MLP median max ~39); only `clip_grad_norm_(1.0)`
   + the trust-region keep it stable. With 2× the parameters and far spikier gradients, PNA is strictly
   more expensive with zero benefit here.

## Honesty / scope
- 5 seeds × {random, urban}, single config (the recommended residual+PBRS+trust-region). The exact-0
  paired diff means both actors are clamped to the anchor by the trust-region — neither learns a
  beneficial deviation, consistent with the Q7/Q8 ceilings and Q9 PART 2.
- This does NOT prove PNA is useless in general — only that in THIS residual frame it equals MLP equals
  the anchor. PNA stays OPT-IN.

## Acceptance table (Contract v3 §15)
- Phase: **Q11 — PNA in the residual framework**
- Status: **VALIDATED_NEGATIVE** (PNA == MLP == anchor; D13 trend does not reproduce; no over-claim).
- Implemented ✓ / Wired (`--actor pna` in the residual trainer) ✓ / Active in this run ✓ (5 seeds × 2 arch)
- Test scale: 1 new unit (PNA drop-in) + suite 770/0; 5 seeds × {random, urban} × {mlp, pna}
- Positive: PNA trains stably in the residual frame (no collapse, no NaN) — the trust-region tames the D13 instability.
- Negative: PNA gives NO feasibility advantage over MLP/anchor (paired diff exactly 0); 2× params; gradient-explosive.
- Conclusion scope: in the residual+PBRS+trust-region frame, PNA == MLP == anchor at N≤16; the PNA trend
  is not real here. NOT a claim about PNA at other scales.
- Next action: **Q12 — the multi-seed / CSI-mode / N campaign** (all arms — anchor / imitation / add-repair
  / prune / full-residual / +PBRS / +PNA / central-myopic-ref — ≥5 seeds, per-seed + CI, urban + random).

## Adversarial verification (multi-lens Workflow `wslezjbo1`, 4 lenses) — overall MINOR, no blocker/major
- **FAIR COMPARISON — PASS.** The sole difference is `make_actor` (PNA vs MLP); residual action space,
  anchor, trust-region (0.5), PBRS, eval-no-shaping, train/held scenes (seed×1000+1 / +777), feature
  standardization, sampling RNG (seed+1), budget (20 updates, lr 0.02), and the logit head/decode are all
  shared and actor-independent. **Anchor metrics are byte-identical across arches at every seed×data**
  (proves the same scenes); independent same-flags reproduction matched exactly. No confound.
- **CI / SEED-COLLAPSE — MINOR.** Recomputed from all 10 raw JSONs (t.975(df=4)=2.776): paired
  (residual−anchor) = **+0.000000, CI [0,0]** for both arches on both data; **0/5 collapse** both;
  PNA residual feasibility element-wise IDENTICAL to MLP (no D13 +0.125). MINOR: my "MLP ~255" was the
  random-data max — MLP's global grad max is 3054.8 (urban s4) — **fixed above** (PNA still far spikier).
- **NO PNA OVER-CLAIM — MINOR.** 20-arm scan: NO seed where PNA (or MLP) residual feasibility > anchor —
  nothing suppressed; the convergence-to-anchor is a genuine learned trust-region outcome (logits>3
  reachable), not a structural rig. Same "MLP ~255" wording nit (fixed). The honest conclusion (PNA gives
  no advantage) is fully supported.
- **PNA GENUINELY ACTIVE — PASS.** PNA is 2.046× the params and has the real directional-PNA aggregation
  + GRU + message passing (not a silent no-op / MLP-equivalent); gradients reach its PNA-specific layers.

**Verdict: KEEP — the honest negative stands. PNA == MLP == anchor in the residual frame; PNA stays
opt-in. The only correction was the cosmetic "MLP ~255" gradient figure (now ~3055 global), applied above.**
