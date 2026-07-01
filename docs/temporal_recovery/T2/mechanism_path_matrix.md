# T2 — Mechanism-Path Matrix (Contract v4 §2)

Status ∈ {NOT_PRESENT, IMPLEMENTED_ONLY, CALLABLE, ACTIVE_IN_LOSS, ACTIVE_IN_EVAL, ACTIVE_IN_DEPLOY}.

| mechanism | code_path | status @ T2 | evidence |
|---|---|---|---|
| leaky-tanh residual activation (`residual_leak`) | `models/belief_residual_actor.py` forward | ACTIVE_IN_EVAL (opt-in; campaign leak=0.1) | `z=s·tanh(raw/s)+leak·raw`; `test_actor_applies_leaky_map` |
| non-vanishing residual gradient | same | ACTIVE (leak>0) | `test_leaky_map_gradient_never_vanishes` (grad ≥ leak at |raw|=100) |
| tanh saturation (T0 defect) | same | RETAINED at default leak=0 (back-compat) | `test_residual_logit_saturates_and_gradient_vanishes` (leak=0 still saturates) |
| raw-logit L2 penalty (R1) | `training/residual_saturation.py` | ACTIVE_IN_LOSS (unchanged) | still the trained-regime `raw` regularizer |
| recurrent→logit signal (effect-on-decision) | forward + `recurrent_vs_memoryless_delta` | ACTIVE_IN_EVAL | leaky ≥ tanh signal (pilot + saturation-regime test) |

**Blast radius: zero at the default.** `residual_leak=0.0` (default) reproduces the frozen R1–R8 tanh head
byte-identically — the completed Belief-Residual campaign and its tests are untouched. The Temporal-Recovery
campaign's actor (T3+) constructs with `residual_leak>0`. No other module changed.
