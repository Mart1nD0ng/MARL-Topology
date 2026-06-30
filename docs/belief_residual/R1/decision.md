# R1 — decision (feature standardization + logit-saturation fix)

**Result: KEEP.** The Q14 "recurrence is behaviorally inert" was caused by the residual path reusing the
shared ±10 tanh head (saturated `raw`≈3000). A **separate small-range residual head** (`z = 3·tanh(raw/3)`) +
**raw-logit L2** + **all-train-frame standardization** fixes it: under stale CSI the recurrent vs memoryless
residual LOGITS are no longer bit-identical and the head is unsaturated. **Honest scope: the fix is at the
LOGIT level; the final action/topology is unchanged (action_delta=0) — that is R3.** No "recurrence
ineffective" claim is permitted now (Contract v4 §8/§10 satisfied). Verification: Workflow `wfv50jg36`.

## What changed (one variable: the residual logit head + its input scale; additive)
- `src/marl_topology/models/belief_residual_actor.py` — `BeliefResidualActor`: the proven node-encoder +
  directional message passing + per-node `GRUCell`, with a **separate residual head** outputting
  `z_max·tanh(raw/z_max)` (`z_max=3`) and exposing `raw`. (Belief/repair/safety heads = R2/R5 placeholders.)
- `src/marl_topology/training/residual_saturation.py` — `feature_standardization_all_frames`,
  `saturation_metrics`, `raw_logit_l2_penalty`, `recurrent_vs_memoryless_delta`.
- Does NOT modify `DynamicRecurrentActor` or `residual_pbrs_train.py` (the R0 PPO-spy tripwire still holds).

## Effect-on-Decision (pilot, urban delay-1, after identical logit-pushing optimization)
| metric | OLD ±10 shared head | NEW ±3 separate head + raw-L2 |
|---|---|---|
| frac_logit_near_rail | **1.0** (saturated) | **0.0** (unsaturated) |
| raw_abs_mean | 72.5 | 4.1 |
| recurrent−memoryless **logit** delta | **0.0 (INERT)** | **0.0257 (passes)** |
| recurrent−memoryless **action** delta | 0.0 | **0.0** |

The old head reproduces Q14's saturation (fully railed → recurrence inert); the new head is unsaturated and
the GRU's cross-frame signal reaches the logits. **But the logits differ too little to flip any edge's MAP
decision yet — `action_delta=0`.** Per Contract v4 §4 this is reported, not hidden: R1 fixes the channel for
the temporal signal; whether that signal changes the topology depends on a trained policy operating near
decision boundaries (R3) + the evidence-gated candidate set (R6). (Verification noted an independent untrained
probe reached `action_delta`≈0.07 in another config, so the pilot's 0.0 is a conservative point estimate.)

**Essential nuance (do not over-attribute to z_max): the fix is the COMBINATION of the small z_max AND the
raw-logit L2 — z_max alone is NOT sufficient.** Verification confirmed that with `lam_raw=0`, the ±3 head
ALSO fully saturates (`frac_logit_near_rail`=1.0, `raw_abs_mean`~135) — a bounded tanh still rails when the
objective pushes `raw` large. The raw-L2 penalty is what keeps `raw` small (organically ~3–4); the small
z_max alone only caps the rail value, not the saturation. R3 must therefore keep raw-L2 ACTIVE_IN_LOSS.

## Path-specific scope (Contract v4 §5)
> "The separate small-range residual head + raw-L2 removes the LOGIT-level saturation that made recurrent ==
> memoryless under stale CSI (urban delay-1, single pilot). It does NOT by itself change the final topology
> (action_delta=0); that is the question R3 (residual PPO trainer) answers."

## Tests (failing-first → pass; load-bearing + Effect-on-Decision)
5 in `tests/unit/test_belief_residual_R1_saturation.py` (all fail on HEAD: modules absent → pass after
implementation). Full suite **780/0** (775 + 5). The Effect-on-Decision test asserts a real recurrent−
memoryless logit difference + an unsaturated head; the raw-L2 test asserts L2 actually shrinks `raw`.

## Acceptance (Contract v4 §15)
- Claim Cards ✓ / Mechanism-Path Matrix updated ✓ / Load-bearing + Effect-on-Decision tests pass ✓ (5/5,
  suite 780/0) / activation artifact ✓ / pilot with raw artifact ✓ / decision ✓ / scope explicit ✓.
- Exit condition (Workflow R1) MET: saturation rate dropped (1.0→0.0) AND recurrent vs memoryless logits no
  longer bit-identical. Action-delta=0 is honestly recorded as the LOGIT-vs-ACTION gap, deferred to R3.
- Decision: **KEEP** — proceed to R2.
- Next: **R2** — CSI belief prediction auxiliary (`belief_head`, `L_CSI`; true current psucc = training label
  only; recurrent belief MSE < memoryless under delay; belief loss enters the policy training loss).

## Adversarial verification (Workflow `wfv50jg36`, 4 lenses) — overall MINOR, no blocker/major
- **SATURATION GENUINELY DROPPED — MINOR.** Reproduced: new head `raw_abs_mean` 3.3 / rail 0.0 vs old 228 /
  rail 1.0; `|logit|≤z_max` holds by construction; `raw_logit_l2_penalty`=`mean(raw²)` correct; metric
  scale-relative. MINOR = the fix is **z_max AND raw-L2 jointly** (z_max alone also rails) — folded in above;
  the artifacts already attribute it to the combination, so nothing was over-stated.
- **RECURRENCE GENUINELY PASSES — PASS.** Decisive control: both arms `hidden=None` → delta **exactly 0.0**;
  carried hidden → **0.02571** (reproduced to 5 dp), backed by a 0.4666 hidden divergence growing
  frame-by-frame from `dh=0` at t=0. Old ±10 head reproduces Q14's zero-delta inertness. Not noise/bug.
- **ACTION-DELTA = 0 HONESTLY SCOPED — PASS.** `action_delta=0.0` is a top-level field for both heads (not
  hidden); the pilot reproduces byte-identically; the exit condition + tests + docstrings scope R1 to the
  LOGIT level and defer topology change to R3. No over-claim.
- **ADDITIVE + NO REGRESSION — MINOR.** `git diff` confirms `dynamic_recurrent_actor.py` + `residual_pbrs_
  train.py` byte-identical to HEAD; `BeliefResidualActor` is a standalone module; only the status doc was
  modified. The 5 tests are load-bearing (mutation probes collapse the recurrent-vs-memoryless test to 0 when
  the cross-frame signal is removed). Suite 780/0.

**Verdict: KEEP — R1 stands; the honesty nuance (z_max+raw-L2 jointly) is recorded; proceed to R2.**
