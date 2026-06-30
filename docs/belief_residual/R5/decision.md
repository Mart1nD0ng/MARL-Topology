# R5 — decision (repair/safety/edit heads) — the DEPLOYABLE-LEARNING crux

**Result: KEEP — PARTIAL POSITIVE (the campaign's first deployable-LEARNING positive — decisive on random
at n=5).** Trained on ONLY LOCAL deployable features (the actor's graph encoding of nf/ef + anchor membership —
**no evaluator, no critic, no central/global decoder**; see the CSI-scope note below for what "no true CSI"
means here), the edit head's held top-k beneficial-edit precision **beats the same-budget random baseline on
random data (5-seed CI entirely > 0, lower bound +0.089), and in mean (4.5× lift, but not 95%-significantly at
n=5) on urban.** Local features CAN predict the R4 central beneficial edits → **KEEP → R6** (gate the residual
action on the heads), with the urban-significance gap and the n=5 small-sample fragility stated as the scope
boundary.

## CSI / observation scope (R5 runs with the stale-CSI overlay OFF — what "no true CSI" means here)
R5 (like the R3/R4 residual track) runs with the **stale/partial-CSI POMDP overlay OFF** — the R5 generator
(`residual_ppo_train._build`) passes no `csi_observation_model`, so the per-edge feature `ef` carries the
**current-frame** link channel (the deployable channel observation used throughout R3/R4 and the dynamic-RL
trunk), NOT a stale observation. So in R5, **"no true CSI" means the head input never sees the
EVALUATOR/critic/central reliability computation** (the load-bearing decentralization constraint — verified
leak-free), NOT that the channel is stale. The stale-CSI axis is the SEPARATE opt-in dimension of R1/R2 (shown
to be a no-op for recovery). Consequence for scope: R5 proves local-feature learnability of the R4
beneficial-edit ranking **on the current-frame channel**; whether the learned ranking helps under the stale-CSI
deployment regime (the campaign's actual failure axis) is evaluated end-to-end at R6/R8 with the overlay on.

## 5-seed result (`edit_heads_metrics.json`, powered: train 5 scenes × build, held 10 scenes × build, 4 frames)

| metric | random | urban |
|---|---|---|
| held top-k precision | 0.425 [0.239, 0.611] | 0.324 [0.158, 0.490] |
| random base rate | 0.174 [0.128, 0.220] | 0.140 [0.110, 0.170] |
| **precision − base (paired 95% CI)** | **+0.251 [+0.089, +0.413]** ✅ CI>0 | +0.184 **[−0.005, +0.373]** ⚠ spans 0 by 0.005 |
| lift (precision / base) | 3.17× | 4.50× |
| repair_corr (held ΔD-reduction) | **+0.226 [+0.176, +0.277]** ✅ | +0.178 [−0.085, +0.440] |
| safety_corr (held deletion-risk) | +0.105 [−0.208, +0.419] | +0.058 [+0.005, +0.111] ✅ |
| **untrained control** (precision − base) | +0.003 [−0.088, +0.094] | +0.030 [−0.033, +0.093] |

**Read (honest, both terms reported):**
- **Random — significant at n=5.** `precision − base` CI is entirely above 0 (lo +0.089; 3.2× lift) — though
  the lower bound is not far from 0 and a single reproduced seed gave +0.324 vs the 5-seed mean +0.251, so the
  n=5 CI is fragile (call it "decisive at n=5", not regime-invariant). `repair_corr` CI is also entirely above
  0 ([+0.176, +0.277]) — the headline rests on edit precision + repair_corr, NOT on safety_corr (which spans 0
  on random, [−0.208, +0.419], and is reported but not load-bearing). The **UNTRAINED** head's `precision −
  base` sits at chance ([−0.088, +0.094]) on the same held frames → the lift is produced by *training the heads
  on local features*, NOT a metric artifact or topology prior (a causal ablation confirms it: zeroing only the
  trained `edit_head` collapses held `precision − base` from +0.324 to −0.128). This is the first time in the
  campaign that a deployable (local-only) head has learned the beneficial-edit direction — the exact piece R3
  diagnosed as missing ("residual==anchor = no direction signal").
- **Urban — strong mean, not yet significant.** Mean `precision − base` +0.184 (4.5× lift, precision 0.32 vs
  base 0.14), 4/5 seeds positive, but the 5-seed t-CI lower bound is −0.005 — a hair below 0. `safety_corr` CI
  is positive though small. So the signal is clearly present in mean but not 95%-significant at n=5 (conservative
  t with 4 dof; could tighten with more seeds). The urban deployable gain is NOT claimed as proven.

This is **NOT** a deployable-learning gap (the strong-negative outcome R5 was set up to detect): on random the
heads decisively learn the signal, and on urban the mean lift is even larger. It is a PARTIAL positive whose
only caveat is urban 95%-significance at n=5.

## What changed (additive heads on the BeliefResidualActor; supervised on R4)
- `edit_head` / `repair_head` / `safety_head` appended to `BeliefResidualActor` (after the R2/R5 heads, init
  RNG preserved); `edit_scores(nf, ef, ei)` returns per-edge {edit_logit, repair_pred, safety_pred} from the
  per-edge LOCAL features `[ef, h_u⊙h_v, |h_u−h_v|]` — no evaluator / no true CSI in the head input path.
- `src/marl_topology/training/edit_head_training.py`: `build_edit_examples` (per-frame local features +
  per-candidate-edge targets from the R4 evaluator scoring — TRAINING LABELS only), `train_edit_heads`
  (`L_edit` BCE + `L_repair`/`L_safety` Huber; SEPARATE held eval), `topk_metrics` (held precision@k vs random
  base; the untrained-head control isolates "is it learning or is it the metric?").

## Effect-on-Decision / metrics
The heads change the **ranking** of candidate edits (held top-k precision +0.25 random / +0.18 urban over the
same-budget random baseline). Whether that learned ranking converts into a deployable feasibility/return gain at
DEPLOYMENT (gating the residual action on the heads, 0 evaluator calls) is the R6 question — R5 proves the
ranking is locally learnable, NOT that it improves the deployed topology yet.

## Path-specific scope (Contract v4 §5)
> "Trained on LOCAL features only (deployable — head input sees no evaluator/critic/central decoder; ef is the
> current-frame channel, stale-CSI overlay OFF), the edit head's held top-k beneficial-edit precision BEATS the
> same-budget random baseline on random (5-seed CI [+0.089, +0.413], 3.2× lift, untrained control at chance —
> significant at n=5 though the CI lower bound +0.089 is small) and IN MEAN on urban (mean +0.184, 4.5× lift,
> but 5-seed CI [−0.005, +0.373] spans 0 by 0.005, i.e. NOT 95%-significant at n=5), at N≤16. The R4
> beneficial-edit signal — proven to exist — IS locally learnable on the current-frame channel. This is the
> deployable-LEARNING question answered POSITIVELY (significant on random, mean-positive on urban); it is NOT a
> learning gap. Open boundaries: (a) urban 95%-significance (n=5); (b) whether the learned ranking yields a
> deployed feasibility/return gain (R6); (c) whether it survives the stale-CSI overlay (R6/R8, the campaign's
> actual failure regime). The headline rests on edit precision + repair_corr (CI>0), NOT safety_corr (spans 0)."

## Tests (failing-first → pass)
5 in `tests/unit/test_belief_residual_R5_heads.py` (fail on HEAD: module/method absent → pass). The heads are
local-only (signature audit + no-evaluator run); the top-k metric is verified correct (precision>base on good
ranking, <base on bad). The empirical "beats random?" question is the DECISION (measured by the 5-seed),
deliberately NOT asserted by a unit test (which would prejudge the honest result). Full suite **805/0**.

## Acceptance (Contract v4 §15)
- Claim Cards ✓ / Mechanism-Path Matrix ✓ / load-bearing tests pass ✓ (5/5, suite 805/0) / activation artifact
  ✓ (`mechanism_activation.json`) / 5-seed raw ✓ (`edit_heads_metrics.json`) / decision ✓ / scope explicit ✓.
- Exit condition (per-regime, not jointly): held top-k precision vs random — KEEP if EITHER regime's
  `precision_minus_base` CI > 0 with the untrained control at chance (→R6); STOP only if NEITHER regime beats
  random (deployable learning gap). **Random CI>0 at n=5 (lo +0.089) with untrained control at chance and
  repair_corr CI>0; urban CI spans 0 by 0.005 but mean strongly positive (4.5×) → gate MET on the random regime
  → KEEP → R6** (local learnability established; urban significance + stale-CSI survival flagged as scope
  boundaries, to be resolved by R6's end-to-end deployed eval on BOTH regimes with the overlay on).
- Decision: **KEEP → R6.**

## Adversarial verification (Workflow) — see below; 5-seed CI — `edit_heads_metrics.json`
