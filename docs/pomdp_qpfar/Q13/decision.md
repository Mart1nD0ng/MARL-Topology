# Q13 — decision (docs / report / README close-out) — CAMPAIGN COMPLETE

**Result: KEEP. The POMDP-QP-FAR campaign (Q0–Q13) is consolidated into the durable docs, every headline
traceable to a committed artifact, no over-claim.** This pure-docs stage closes the campaign. Adversarial
verification: claims-vs-evidence Workflow `wongcg9aj` (verdict below).

## What changed (one variable: the documentation)
- `docs/URBAN_V2X_RESEARCH_LOG.md` — appended the **POMDP-QP-FAR CAMPAIGN SUMMARY** (the two-axis problem,
  the Q0–Q13 stage ledger with commits, the Q12 central result, the mechanism ledger, the v2/D-consistency).
- `docs/CURRENT_HEAD_STATUS.md` — prepended a **POMDP-QP-FAR CAMPAIGN COMPLETE** banner (central result +
  recommended config + scope + the Workflow ids).
- `docs/CURRENT_POMDP_QPFAR_STATUS.md` — Q13 row DONE + a **🏁 ENTIRE Q0–Q13 COMPLETE** section.
- `README.md` — added the third (POMDP-QP-FAR) campaign to the headline-result section.
- `AGENTS.md` — prepended a current-state pointer (three campaigns; the opt-in POMDP-QP-FAR mechanisms;
  the deployable anchor as the best deployable arm; the central myopic as a grouped ceiling).

No `src/` or `tests/` change — the suite stays **772/0** (`git diff --stat` shows only `.md` files).

## The consolidated campaign result (recorded)
The POMDP-QP-FAR campaign attacked the two D0–D14 bottlenecks (Axis A temporal degeneracy via stale/partial
CSI; Axis B feasibility plateau via `D_quorum` + feasible-anchored residual learning). Every mechanism is
verified-correct, opt-in, default byte-identical, and individually informative (Q2/Q4 gates, Q7 22%-repair,
Q8 safe-prune, Q10 −33% urban disagreement). The Q12 5-seed headline: the DEPLOYABLE tier (anchor ==
residual ± PBRS ± PNA) sits at urban 0.739 / random 0.281 with 0 evaluator calls, below the central oracle
(0.800 / 0.308) IN MEAN but the paired CI spans 0 (not significant; 1/5 seeds the anchor wins). **No learned
arm beats the deployable anchor — feasibility-region learning is the binding limit**, the THIRD independent
campaign to reach this finding.

## Honesty / scope
- Pure-docs; per Contract v3 a docs-only stage's verification is the claims-vs-evidence audit (below) + the
  suite staying green with no executable change.
- Every headline number is traceable to a committed decision/JSON + a commit hash. Scope: N≤16, 5 seeds,
  urban + random; N≥24 is the labelled open frontier.
- The push to origin is the OWNER's decision — surfaced via AskUserQuestion after this commit, NOT performed
  autonomously.

## Acceptance table (Contract v3 §15)
- Phase: **Q13 — docs close-out (campaign complete)**
- Status: **DONE** — five docs consolidated, internally consistent, every claim traceable, no over-claim.
- Implemented ✓ (docs) / Wired (n/a) / Active (n/a — pure docs)
- Test scale: suite 772/0 unchanged (no src/test drift); claims-vs-evidence Workflow `wongcg9aj`.
- Positive: the campaign is durably recorded; the recommended config + scope + open frontier are explicit.
- Negative: n/a (docs).
- Conclusion scope: the whole Q0–Q13 campaign, scoped to N≤16, 5 seeds, urban + random.
- Next action: **AskUserQuestion** — surface the push decision to the owner. Then loop idle / awaiting owner.

## Adversarial verification (claims-vs-evidence Workflow `wongcg9aj`, 4 lenses) — overall MINOR, no blocker/major
- **NUMERIC CLAIMS — PASS.** Every headline number reproduces from its source: Q12 urban 0.739 / random
  0.281 (anchor), 0.800 / 0.308 (myopic), 0 vs 432 eval calls, paired CIs urban +0.061 [−0.019,+0.141] /
  random +0.028 [−0.031,+0.086] (recomputed from `q12_campaign.json` per-seed), 1/5-seed crossover; Q11
  paired 0.000; Q7 22%/+0.36; Q10 −33%; Q4 Spearman 0.69–0.95; suite 772/0; all 14 commit SHAs match git.
  (Only nit: the urban CI lower bound prints −0.019 vs exact −0.01848 — a rounding inherited from the Q12
  source, immaterial; the interval still spans 0.)
- **NO OVER-CLAIM / GROUPING — MINOR.** None of the five prohibited statements appears (no central ref
  called deployable; no learned-beats-anchor; gap consistently stated not-significant / paired CI spans 0;
  PNA +0.125 explicitly does NOT reproduce; nothing extrapolated past N≤16). MINOR: the banner subtitle
  read rhetorically positive — **fixed** to "confirms the same honest negative."
- **SCOPE + COMMIT TRACEABILITY — PASS.** Scope N≤16 / 5 seeds / urban+random with N≥24 as the labelled
  open frontier in all five docs; all 14 cited commits exist with matching messages; residual==anchor
  attributed to Q11 (5-seed); `git diff --stat` shows only the 5 `.md` files (pure-docs).
- **CROSS-DOC CONSISTENCY — MINOR.** No genuine contradiction on any dimension (central result, config,
  grouping, suite count, scope, push status all agree). MINOR: the new banner/README/AGENTS sections
  omitted the "unpushed — owner's decision" line carried by the status/log docs — **added** to the banner.

**Verdict: KEEP — the docs close-out is faithful and complete; the two cosmetic MINORs are fixed.**
