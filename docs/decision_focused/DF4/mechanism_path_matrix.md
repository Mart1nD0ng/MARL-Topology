# DF4 — Mechanism-Path Matrix

| Mechanism | State | DF4 finding | Path |
|-----------|-------|-------------|------|
| Leaky-tanh per-edge residual head | CALLABLE (opt-in `residual_leak`, T2; default byte-identical) | sound, near-optimal (bounded + non-vanishing gradient); KEEP | unchanged |
| Softmax over per-edge logits | NOT_PRESENT | rejected: no-op at deployment (top-b == sigmoid) + N-dependent in training (`1/N`) — category error | not built |
| Edit-selection (softmax/Gumbel over candidate edits) | NOT_PRESENT | the softmax-appropriate mechanism, but blocked by the aleatoric wall (direction below-chance on decision-critical edges, DF3d; belief-injection == echo, DF2/DF3) | not built (would hit the same wall) |

## Effect on the decision
DF4's Test 1 is precisely an effect-on-decision result: softmax's acted top-b is **identical** to the deployed
sigmoid/psucc top-b (0.9935) — i.e. softmax changes the forward representation but NOT the final topology. So
softmax cannot help the deployed decision by construction; its only effect is a (wrong) training gradient.

## No deployed-path change
DF4 adds one diagnostic (`df4_activation_probe.py`). The deployed actor keeps the leaky-tanh residual head (opt-in,
default-off). Nothing new is ACTIVE_IN_DEPLOY.
