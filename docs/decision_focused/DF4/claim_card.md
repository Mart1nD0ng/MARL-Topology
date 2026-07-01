# DF4 — Claim Card

**Claim:** For the per-edge residual logit, **leaky-tanh is a sound, near-optimal activation** (bounded +
non-vanishing gradient; the T2 fix) and **softmax is the wrong tool** — it is (a) a no-op at deployment (its
top-b == the sigmoid/psucc top-b) and (b) N-dependent in training (per-edge keep-mass `1/N`), a category error for
independent per-edge keep/drop. Softmax is correct only for a *different* mechanism (edit-selection), which the
campaign shows is itself blocked by the aleatoric wall — so the activation/head choice is moot at feasibility.

**Path covered:** goal 3 (activation design question). Answered.

**Evidence:**
- `df4_activation_metrics.json`: Test 1 argmax-equivalence — softmax top-b == anchor **0.9935**, sigmoid top-b ==
  anchor **0.9935** (928 node-frames; the residual is ties). Test 2 N-dependence — softmax per-edge keep-mass
  `{0.50, 0.25, 0.125, 0.0625, 0.042}` for `N={2,4,8,16,24}` vs sigmoid `0.5`.
- DF0 research lens C (the analytical argument): bounded + non-vanishing-gradient properties; softsign is a
  downgrade; small-scale-linear+penalty is the only serious rival.
- T2 (prior campaign, committed): leaky-tanh restored the recurrent signal to the acted logit; byte-identical off.
- DF2/DF3: the edit-selection mechanism (where softmax is right) can't beat the anchor — direction below-chance on
  decision-critical edges (DF3d), belief injection == stale echo (DF2/DF3).

**Certainty:** Test 1 is a monotonicity identity; Test 2 is `softmax(0-vector) = 1/N` — both analytically certain
and empirically confirmed. No adversarial Workflow required (not a contestable inference).

**Falsify:** a softmax head whose deployed top-b differs from the sigmoid top-b (it cannot — monotone); or an
N-invariant softmax per-edge keep-mass (it is `1/N` by construction).

**Scope:** the per-edge residual mechanism of `BeliefResidualActor`. The edit-selection alternative is analyzed +
tied to the campaign's aleatoric finding, not separately retrained (it would hit the same wall).
