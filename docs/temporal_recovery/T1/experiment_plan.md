# T1 — experiment plan (oracle upper-bound: the CSI-recovery ceiling) — THE GATE

## Question
Is the stale-CSI feasibility drop (R8: urban 0.165 / random 0.055) **realizably recoverable** from stale t−1 +
leak-free geometry, or reachable only by a clairvoyant oracle that reads the true current channel? This decides
whether the temporal redesign (T2–T5) is worth building.

## Design — 4 anchor arms (the anchor decides purely on psucc col 0)
For each held scene/frame, decode the **same** `local_hysteresis` anchor but feed it a different psucc column;
score the produced topology with the **true** evaluator (current real channel + closed-form quorum tail). 0 eval
at decision.

- **A floor** — stale psucc (the deployed regime; = R8 `stale_anchor`).
- **B current-belief** — psucc = `sigmoid(belief_logit)` from the *current* R2 direct-`p_t` head (retrained on
  stale features, true-current label training-only). Where today's model sits.
- **C physics-recovery** — psucc from a small **leak-free** regressor `current_psucc ≈ g(stale_psucc, rel_vel,
  distance, Δdistance)` (a physical-residual predictor; the realizable-information ceiling). Trained on train
  scenes, evaluated on held. NO true CSI at inference (only stale + geometry a node broadcasts).
- **D oracle ceiling** — true current psucc (clairvoyant; = R8 `current_anchor`).

## Metrics (per seed, 5 seeds × {random, urban}, 95% CI)
- Feasibility of A/B/C/D; **ceiling = D − A**; **realizable recovery = C − A**; **model gap = C − B**.
- **MSE landscape** (psucc probability space, held): stale-echo MSE (= A's error) vs physics-regression MSE
  (C) vs current-belief MSE (B) vs 0 (D). `belief_mse` / `stale_echo_floor_mse` reused from `csi_belief.py`.
- Return, C/E/L, edit_rate (arms differ only in psucc, so decode is anchor-shaped — report anyway).
- Budget: evaluator calls (training-time only; 0 at decision), wall-clock.

## Decision rule (Contract v4)
- **C − A CI > 0** (physics recovers real feasibility) AND C < D → **realizable headroom exists → KEEP, proceed
  T2–T5** (chase B→C with the correction target / edge state / uncertainty).
- **C − A spans/below 0** (even physics can't beat stale) → recovery is not realizable from geometry → first
  re-check the oracle/regressor design (is the physics model reasonable? are the leak-free features wired?);
  if confirmed, **pivot** per task-1 fallback to adding leak-safe **temporal hidden features** to the env.
- Report **B − A** honestly regardless: it re-confirms (or not) that the *current* model echoes stale.

## Load-bearing tests (failing-first)
- `oracle_injects_recovered_psucc_into_anchor` — spy that arm C/B replace col 0 before `_anchor`, and that
  arm A/D use stale/true respectively; arms produce DIFFERENT topologies when psucc differs.
- `physics_regressor_is_leak_free` — arm C's inference reads only stale ef + geometry, never `context(t)`
  true psucc (the true psucc is used only as the training label).
- `ceiling_equals_R8_stale_drop` — arm D − arm A reproduces the R8 stale-drop sign/scale (cross-check).

## No-substitution / scope
This tests **CSI recovery feeding the anchor**, NOT a learned policy beating the anchor (the prior campaign
exhausted that). A positive here means "recovering the current channel improves the anchor's own decisions",
which is the lever the prior campaign never pulled. Distinct from the anticipation oracle (forecasting, +0.000).
