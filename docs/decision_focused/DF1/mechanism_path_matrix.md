# DF1 — Mechanism-Path Matrix

| Mechanism | State at HEAD `735e5a7` | State after DF1 | Notes |
|-----------|-------------------------|-----------------|-------|
| `shadow_decorrelation_distance_m` knob | NOT_PRESENT (hardcoded 10 m constant) | **CALLABLE** (config-threaded `ChannelModelConfig` + `PhysicsRegime`; swept) | default 10 m ⇒ byte-identical; validated `>0` |
| Shadow-fading field temporal correlation | ACTIVE_IN_EVAL (fixed 10 m cell) | ACTIVE_IN_EVAL (tunable cell) | evaluator/true-channel side only |
| Knob → actor observation | — | **NOT_PRESENT (leak-free by construction)** | knob is channel-side; the staleifier that builds the actor obs never sees the channel config or the shadow field |

## Effect on the decision (not just forward)
DF1 changes the TRUE channel the evaluator scores (via the shadow field cell size), which changes the true
psucc and hence, in principle, the anchor's topology under that channel. DF1 does **not** change the actor's
decision *given a fixed observation* — the actor never sees the knob. The downstream effect on
recovered-vs-anchor feasibility is measured in DF2 (the oracle re-gate), not asserted here.

## Leak-path invariants (DF1 status)
- **Knob is leak-free by construction:** it is a `ChannelModelConfig` field consumed by `evaluate_channel`
  (evaluator/context), never by the actor-observation builder. Asserted structurally.
- **Deferred to DF2 (built with the predictor arms):** the statistical leak battery
  (incremental-R²-over-distance G2, boundary-restricted G3, per-scene G4, residual-shadow G1) and the
  `motion_features=False` enforcement (`dynamic_frames` `csi_delta` uses current psucc — a pre-existing latent
  leak any goal-2 run must close). These test the OBSERVATION pipeline, which DF1 does not modify.

## Byte-identity
`shadow_decorrelation_distance_m` defaults to the 10 m constant; `_shadow_field_value`'s `decorr_m` defaults to
the same; existing callers (no `decorr_m` arg) are unchanged ⇒ every channel record at the default is bytewise
identical to HEAD (existing 37885 suite 10/10 + `test_default_decorr_arg_matches_module_constant`).
