# T2 — experiment plan (non-saturating activation; task 2)

## Hypothesis (one variable: the residual activation)
The residual decision logit `z = z_max·tanh(raw/z_max)` saturates — its gradient `sech²(raw/z_max)` vanishes for
`|raw| ≫ z_max`, so the recurrent temporal signal cannot reach the ACTED logit once `raw` grows (T0 defect 2;
R1 only mitigated this indirectly with a raw-L2 penalty that keeps `raw` small). Adding a **linear skip**
`z = z_max·tanh(raw/z_max) + residual_leak·raw` bounds the gradient below by `residual_leak > 0`, so it never
vanishes and the temporal signal always reaches the logit — without depending on the raw-L2 band-aid.

## Controlled variable
`residual_leak` on `BeliefResidualActor` (default 0.0). `leak=0` is byte-identical to the frozen R1–R8 tanh
head (zero blast radius); the Temporal-Recovery campaign constructs the actor with `leak>0` (0.1). Nothing else
changes (same encoder / MP / GRU / head weights / raw-L2).

## Evidence
1. **Load-bearing / back-compat** (`test_temporal_recovery_T2_activation.py`): `leak=0` reproduces the pure tanh
   map exactly; `leak>0` the actor's forward emits `tanh + leak·raw` (the skip is on the decision path).
2. **Gradient never vanishes**: `d z/d raw = sech²(·) + leak ≥ leak > 0` at large `raw` (vs pure tanh → 0).
3. **Effect-on-Decision** (Contract v4 §4): in a saturation regime a recurrent-vs-memoryless shift in `raw` is
   SUPPRESSED by pure tanh (`logit_delta ≈ 0`, behaviorally inert) but PRESERVED by the leaky map
   (`logit_delta > 0`). Deterministic, isolates the activation.
4. **Real-data pilot** (`t2_activation_pilot.py` → `mechanism_activation.json`): on real delay-1 stale urban
   scenes, same init, only `leak` differs — the leaky head is non-degenerate and carries ≥ the recurrent→logit
   signal of the tanh head.

## Scope (honest)
T2 is an ENABLING fix (like R1), logit/gradient-level. It does NOT by itself convert to a topology/feasibility
gain (the direction signal is T3; the topology conversion is T6). On the trained actor the raw-L2 keeps `raw`
small so both activations are near-linear there; T2's payoff is ROBUSTNESS (the gradient flows even when the
T3 correction signal makes `raw` grow) and removing the dependence on the raw-L2 band-aid.

## Definition of done
Failing-first tests pass; `leak=0` byte-identical (prior suite green); Effect-on-Decision reported; activation
artifact written; decision.md; scope explicit.
