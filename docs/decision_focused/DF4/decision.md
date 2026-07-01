# DF4 — decision (goal 3: activation — leaky-tanh vs softmax)

**The owner's goal-3 question — "is leaky-tanh the best activation, and why not softmax?" — answered, with a
focused empirical probe grounding the DF0 analysis.**

## Is leaky-tanh the best? — YES for the per-edge residual logit (KEEP it).
The residual head emits a **per-edge scalar logit** added to an independent per-edge keep/drop decision. Two
properties matter: (a) **bounded** output (trust-region / PPO stability — an update can't blow a decision
arbitrarily past the anchor boundary); (b) a **gradient bounded away from zero** so the recurrent/temporal signal
always reaches the *acted* logit. Leaky-tanh `z = z_max·tanh(raw/z_max) + leak·raw` satisfies both by construction
(gradient `= sech²(raw/z_max) + leak ≥ leak > 0`). It is the minimal fix that restored (b) — plain ±10 tanh
saturated (`1−tanh²→0` at the rails) and annihilated the recurrent signal (the T2 defect, already fixed and
committed; byte-identical off). Alternatives: plain tanh / hardtanh saturate (the defect); identity is unbounded
(instability); relu/softplus are one-sided (cannot push a logit negative); **softsign is a downgrade** (its
gradient still → 0). The only serious rival is a **small-scale-linear head + magnitude penalty** (a soft/statistical
bound vs leaky-tanh's hard per-sample cap). Verdict: leaky-tanh is a sound, near-optimal choice; not provably
unique, but no cheaper option restores the non-vanishing gradient without giving up boundedness.

## Why NOT softmax? — a category error for per-edge keep/drop, confirmed empirically.
Softmax normalizes a vector to a distribution summing to 1 — it models a **single mutually-exclusive choice** among
elements. The per-edge residual is **N independent** keep/drop decisions. Two concrete failures, both measured
(`df4_activation_metrics.json`):
- **It changes NOTHING at deployment (Test 1).** softmax over the incident-edge logits is monotone in the logit
  (= monotone in psucc), so its top-b selection is IDENTICAL to the sigmoid/psucc top-b: **softmax top-b == anchor
  0.9935, sigmoid top-b == anchor 0.9935** over 928 node-frames (the ~0.6% gap is ties). So softmax is a no-op at
  the acted argmax — it cannot improve the deployed decision; it only differs in the training gradient (where its
  off-diagonal `−p_i p_j` coupling is wrong for independent edges).
- **It breaks cross-N and multi-edge (Test 2 — the category error).** For N equal-psucc incident edges, the
  softmax per-edge keep-mass is `1/N` `{0.50, 0.25, 0.125, 0.0625, 0.042}` for `N={2,4,8,16,24}`, vs sigmoid/tanh's
  N-invariant `0.5`. Softmax under-connects high-degree nodes and cannot express "keep many edges at once" — fatal
  for a budget-top-k, multi-edge, cross-N decision (the project's whole point).

## Where softmax WOULD be right — and why it is moot here.
Softmax/Gumbel is correct for a DIFFERENT mechanism: an **edit-selection** head that chooses *which* candidate
edit(s) to apply under the node budget (a categorical / top-k / Gumbel policy over an edit set). The DF0 research
flagged this as the high-value alternative because it consumes only the rankable DIRECTION signal. **But the
campaign shows edit-selection cannot help either:** injecting a nowcast belief into the anchor's ranking (which IS
edit-selection) == the stale echo (DF2/DF3 realizable arm, no feasibility gain), and the direction an edit-selector
would rank by is **below chance on the decision-critical edges** (DF3d: dir_acc 0.40 on echo-wrong moved edges).
So the activation/head choice is **moot at the feasibility level**: the anchor is the ceiling of all deployable
arms, and no head — leaky-tanh, softmax, or edit-selection — reaches it, because the binding limit is
information-in-features (goals 1 & 2), not the output parametrization.

## Disposition
- **KEEP leaky-tanh** for the per-edge residual logit (opt-in `residual_leak`; default-off byte-identical). Do NOT
  softmax the per-edge logits (no-op at deployment + N-breaking in training).
- Goal 3 = **answered** (design question): leaky-tanh sound; softmax a category error; edit-selection the
  softmax-appropriate alternative but blocked by the same aleatoric wall as goals 1 & 2.
- No deployed-path change; the empirical claims are a monotonicity identity + an N-dependence fact (analytically
  certain, empirically confirmed) — no adversarial Workflow needed.
- **Next: DF6 close-out + 中文 report.** (DF5 collapses: there is no winning config to run a campaign on — every
  deployable arm equals the anchor; the ≥5-seed evidence is already in DF1–DF3.)
