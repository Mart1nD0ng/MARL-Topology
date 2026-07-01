# DF4 — Experiment Plan (goal 3: activation — leaky-tanh vs softmax)

**Owner goal 3:** is leaky-tanh the best activation, and why not softmax? A DESIGN question — answered analytically
in DF0 (research lens C) and grounded here by a focused empirical probe. Base HEAD `1592cb4` (DF3). No large
experiment / no training / no deployed-path change.

## What DF4 did
- `scripts/diagnostics/df4_activation_probe.py`, two deterministic tests on the real anchor scenes + numerics:
  1. **Argmax equivalence:** softmax over incident-edge logits is monotone in the logit → its top-b selection is
     identical to the sigmoid/psucc top-b (**0.9935** on 928 node-frames) ⇒ softmax cannot change the deployed
     decision.
  2. **N-dependence (the category error):** softmax per-edge keep-mass = `1/N` (0.50→0.042 for N=2→24) vs
     sigmoid/tanh N-invariant `0.5` ⇒ softmax under-connects high-degree nodes / cannot keep many edges.
- `decision.md`: the consolidated goal-3 answer (leaky-tanh sound; softmax category error; edit-selection the
  softmax-appropriate alternative, but blocked by the campaign's aleatoric wall — moot at feasibility).

## Why no adversarial Workflow / no ≥5-seed campaign
The empirical claims are a monotonicity identity (Test 1) + an N-dependence fact (Test 2) — analytically certain,
empirically confirmed; not a contestable mechanism inference (unlike DF2/DF3). The leaky-tanh non-saturation
result is already established + committed (T2, byte-identical off). There is no learned arm that beats the anchor
(DF1–DF3), so a "winning-config" 5-seed campaign (DF5) has nothing to run — it collapses into "the anchor stands;
the ≥5-seed evidence is in DF1–DF3."

## Exit criteria
- [x] `df4_activation_probe.py` built + run (argmax-equivalence 0.9935; N-dependence 1/N).
- [x] `decision.md` + claim_card + MPM (goal-3 answer).
- [ ] DF4 committed (no push); DF6 close-out + 中文 report next (DF5 collapsed).
