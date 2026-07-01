# DF0 — Mechanism-Path Matrix

Path levels: NOT_PRESENT · IMPLEMENTED_ONLY · CALLABLE · ACTIVE_IN_LOSS · ACTIVE_IN_EVAL · ACTIVE_IN_DEPLOY.
DF0 changes NO code; this matrix records the *starting* state of each mechanism this campaign will touch, so later
stages can show the delta they actually move.

| Mechanism | State at HEAD `1bf773f` | Target stage / intended path |
|-----------|-------------------------|------------------------------|
| Shadow-fading field `_shadow_field_value` (temporal correlation) | **ACTIVE_IN_EVAL** (on via `operating_point_regime`; feeds the true channel the evaluator scores) | DF1: expose `d_corr` knob (stays ACTIVE_IN_EVAL; default byte-identical) |
| `SHADOW_DECORRELATION_DISTANCE_M` as a **tunable** knob | **NOT_PRESENT** (hardcoded module constant, not a config field) | DF1: → CALLABLE (config-threaded, swept); never ACTIVE_IN_DEPLOY as an actor input |
| Decision-focused objective (BWAR) | **NOT_PRESENT** | DF3: → ACTIVE_IN_LOSS (training-only; opt-in) |
| MSE / NLL belief objective (`csi_belief.py`) | ACTIVE_IN_LOSS (opt-in belief training) | DF3 baseline arm (unchanged) |
| Nowcaster prediction injected into the deployed anchor | IMPLEMENTED_ONLY (diagnostic oracle arms only; not converting) | DF2/DF3: measured in EVAL; deploy only if it beats the anchor (it has not yet) |
| Edit-selection (top-k / Gumbel-softmax) head | **NOT_PRESENT** | DF4: → CALLABLE, A/B in EVAL |
| Per-edge leaky-tanh residual head | CALLABLE (opt-in `residual_leak`; default byte-identical) | DF4 baseline arm (unchanged) |

**Leak-path invariants asserted by the DF design (to be enforced by DF1 tests, not yet enforced at HEAD):**
- True current CSI / shadow field / seed: **must remain NOT_PRESENT in the actor observation** (ACTIVE only as a
  training label / in the evaluator). The `motion_features` `csi_delta` current-psucc path is a latent leak that
  DF1 must close (set `motion_features=False` or patch to stale) before any goal-2 run.
- Evaluator: **ACTIVE_IN_EVAL on the true current channel only** — never reads `obs["ef"]`
  (`test_evaluator_ignores_ef`).

**Effect-on-decision commitment:** every later stage must report whether its mechanism changes the final
ACTION/topology (decoded via `_mutual`), not merely the forward value — the T-campaign's recurring failure mode
(forward changed, topology bit-identical) is explicitly guarded against.
