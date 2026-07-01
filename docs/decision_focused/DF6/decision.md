# DF6 — decision (campaign close-out) — DF CAMPAIGN COMPLETE

**The Decision-Focused campaign is COMPLETE.** All three owner goals are answered as adversarially-verified honest
negatives that converge on one cause. Pure-docs close-out (no code, no new experiments; suite unchanged 850/0).

## Central result
The owner's three fixes for the "MSE ⊥ decisions" wall — a decision-focused objective (goal 1), an env with
stronger temporal autocorrelation (goal 2), an activation change (goal 3) — are **all honest negatives converging
on ONE cause: deployable precision at the anchor decision boundary is limited by leak-free INFORMATION (aleatoric),
not by the objective, the env temporal structure, or the activation.** The true-CSI oracle converts (paired
`true−mse = +0.131`, p=0.0022) but no realizable arm reaches it.

## Per-goal disposition (all opt-in / default-off; each stage adversarially verified)
- **Goal 1 (DF3):** decision-focused BWAR does not beat MSE (strong hypothesis refuted / aleatoric; small-effect
  underpowered at n=5; B.3(iii) audit: engaged but below-chance). Workflow REVISE → statistics corrected.
- **Goal 2 (DF2):** the `d_corr` knob raises per-link ρ (DF1: 0.044→0.429, marginal-invariant) but does not
  convert (`feas(realizable)==feas(stale)` with a real MSE gain) — self-defeating autocorrelation. Workflow REVISE
  → caveats folded.
- **Goal 3 (DF4):** leaky-tanh sound for the per-edge logit; softmax a category error (top-b == anchor 0.9935 →
  no-op at deploy; keep-mass 1/N breaks cross-N); edit-selection moot.

## What DF6 wrote (docs only)
- `docs/decision_focused/中文总结报告.md` — the owner's task-5 中文 analysis+data report.
- `docs/CURRENT_DECISION_FOCUSED_STATUS.md` — a 🏁 DF CAMPAIGN COMPLETE banner (top).
- `docs/CURRENT_HEAD_STATUS.md` — a 🏁 DF COMPLETE banner atop (the 6th honest negative).
- `docs/URBAN_V2X_RESEARCH_LOG.md` — the `DECISION-FOCUSED (DF) CAMPAIGN SUMMARY` (per-stage DF0–DF4 + commits).
- `AGENTS.md` — DF as the 6th superseding-campaign bullet; "five"→"six" campaigns.
- `docs/decision_focused/DF6/decision.md` — this file.

## Provenance / honesty
Every stage carries an experiment_plan + Claim Card + Mechanism-Path Matrix + a decision card; DF2 and DF3 were
each adversarially verified by a multi-critic Workflow that returned REVISE, and the statistical over-claims were
corrected (DF2 anchored on the point-identity; DF3 downgraded "provably refuted" to the well-powered oracle-gap
claim + the underpowered small-effect caveat, and RAN the missing B.3(iii) engagement audit rather than assert it).
The DF0 validation design was itself adversarially verified before any experiment. No headline rests on a span-0
mean presented as proof; the one well-powered positive (the +0.131 oracle gap) is reported as the paired CI.

## Decision
**CAMPAIGN COMPLETE.** Commits DF0–DF4 (`735e5a7 / e0c59a7 / 40c6962 / 1592cb4 / 731b1b7`) + this DF6 close-out on
`decentralized-marl-trunk`, **NOT pushed** — the push is the owner's decision (asked at close-out). Recommended
config unchanged: deployable arm = `local_hysteresis`; all DF mechanisms (`shadow_decorrelation_distance_m` knob,
BWAR, the diagnostics) opt-in / default-off / byte-identical. **Open frontier (owner's call):** since the wall is
now localized to leak-free INFORMATION, the only promising direction is adding deployable observables (multi-hop
neighbour broadcast, longer stale history, explicit AR shadow-state estimate) under strict leak-free validation —
NOT another objective/env/activation change; and larger-N (≥24).
