# Q13 — documentation / report / README close-out (the campaign's final stage)

## Hypothesis (the single thing this stage establishes)
The Q0–Q12 POMDP-QP-FAR campaign is faithfully consolidated into the durable docs, with every headline
claim traceable to a committed artifact (commit + raw JSON + adversarial-Workflow id), and the project's
top-level docs (research log, head status, README, AGENTS, the QPFAR status) reflect the final state with
NO over-claim. This is a pure-docs stage — no `src/` or `tests/` change; the suite stays green.

## Single change (one variable: the documentation)
Append the POMDP-QP-FAR CAMPAIGN SUMMARY + mechanism ledger to `docs/URBAN_V2X_RESEARCH_LOG.md`; prepend a
POMDP-QP-FAR-COMPLETE banner to `docs/CURRENT_HEAD_STATUS.md`; mark all Q0–Q13 rows DONE in
`docs/CURRENT_POMDP_QPFAR_STATUS.md`; align `README.md` + `AGENTS.md` (the opt-in POMDP-QP-FAR mechanisms,
the deployable anchor as the best deployable arm, the central myopic as the reference ceiling).

## What is recorded (the consolidated, honest headline)
- The campaign attacked the two D0–D14 bottlenecks (Axis A temporal degeneracy; Axis B feasibility
  plateau) with a coherent mechanism stack: stale/partial CSI POMDP (Q1–Q2), quorum-deficit potential
  `D_quorum` + alignment (Q3–Q4), feasible-anchored residual learning + add-repair/prune (Q5–Q8), PBRS
  (Q9), edge handshake (Q10), PNA-in-residual (Q11).
- Every mechanism is verified-correct, opt-in, and default byte-identical. The Q12 consolidated 5-seed
  headline: the DEPLOYABLE tier (anchor == residual == +PBRS == +PNA) sits at urban 0.739 / random 0.281
  with 0 evaluator calls, BELOW the central myopic oracle (0.800 / 0.308) IN MEAN but the paired CI spans
  0 (not significant; on 1/5 seeds the anchor beats the oracle). **No learned arm beats the deployable
  anchor — the binding limit is feasibility-region learning**, confirming the v2/D13 pattern across the
  full POMDP-QP-FAR stack.

## Failing-first test
This is a pure-docs stage. The "test" is the claims-vs-evidence adversarial Workflow (below) + the unit
suite staying green (772/0) with no `src/`/`tests/` change. No new unit test is added (nothing executable
changes); per Contract v3 a docs-only stage is exempt from the failing-test-first rule (its verification is
the claims-vs-evidence audit).

## Success criterion (Q13 passes iff)
1. The five docs are updated and internally consistent; every Q0–Q13 row is DONE.
2. The claims-vs-evidence Workflow finds no over-claim (each headline claim maps to a real commit/JSON/id).
3. `git` shows only doc changes; the suite is still 772/0.

## Failure criterion
Any headline claim not traceable to an artifact; any over-claim (PNA win / deployable-beats-central as
significant / central reference called a deployable baseline / extrapolation past N≤16); any src/test drift.

## Adversarial verification (Ultracode)
A multi-lens Workflow audits the new docs against the raw evidence: (1) every numeric claim matches a
committed JSON / decision; (2) no over-claim or mislabeled group; (3) scope honesty (N≤16, 5 seeds,
urban+random, NOT N≥24); (4) cross-doc consistency (research log == head status == README == QPFAR status).

## Out of scope
The push to origin (owner's decision — surfaced via AskUserQuestion after Q13 commits); N≥24 (open frontier).
