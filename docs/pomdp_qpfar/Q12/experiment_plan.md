# Q12 — Multi-seed / CSI-mode / N campaign (the consolidated POMDP-QP-FAR headline)

## Hypothesis (the single thing this stage establishes)
Across ≥5 seeds × {urban, random} × N∈{8,12,16}, with DEPLOYABLE and CENTRAL-REFERENCE arms grouped
separately (Contract §10.1), the POMDP-QP-FAR deployable arms (anchor + full-residual ± PBRS ± PNA) form
ONE feasibility tier that sits BELOW the central references (myopic-greedy / D_quorum-repair) — i.e. the
learned residual policy MATCHES the deployable `local_hysteresis` anchor (no improvement) and the central
oracle is the ceiling. This is the campaign's consolidated, honest headline.

## Single change (one variable: the consolidation)
A new `scripts/diagnostics/pomdp_qpfar_campaign.py` runs the rollout arms (deployable: local_hysteresis,
local_threshold; central: myopic-greedy) across 5 seeds × {urban, random}, per-seed + CI + evaluator-call
budget. The residual arms (full-residual MLP/PNA ± PBRS) are CITED from Q9/Q11 (5-seed, paired
residual−anchor diff 0.000 — they equal the anchor); the central add-repair (Q7) / prune (Q8) results are
CITED. No new mechanism — this is the consolidation + the grouped headline.

## Two groups (Contract §10.1 — never conflate)
- DEPLOYABLE (0 action-evaluator calls): `local_hysteresis` (the anchor), `local_threshold`,
  full-residual-MLP, +PBRS, +PNA (all == anchor, Q9/Q11).
- CENTRAL REFERENCE (uses the evaluator at action time): myopic-greedy (the oracle), D_quorum-add-repair
  (Q7), conservative-prune (Q8).

## What is measured (workflow Q12) — per seed + CI
per-frame feasibility / episode return / energy / latency / switches / D_quorum / repair-success /
RETENTION / mutual-acceptance / evaluator-calls (the central arms' budget MUST be reported).

## Controlled variables
urban + random; held N∈{8,12,16}; 6 frames; hold_interval=4; γ=0.95; reconfig_e=0.05; ≥5 seeds;
final metrics = true closed-form PBFT C/E/L; deployable arms 0 eval calls.

## Failing-first tests (fail on HEAD)
- `test_campaign_groups_deployable_vs_central` — the driver groups arms into deployable (0 eval calls) vs
  central-reference; the deployable arms have `action_evaluator_calls == 0`.
- `test_campaign_paired_ci_helper` — the per-seed CI helper (mean ± t·se) is correct on a synthetic case.

## Success criterion (Q12 passes iff)
1. Tests pass; the campaign runs ≥5 seeds × {urban, random} and writes a grouped per-seed + CI table.
2. The headline is reported honestly: deployable anchor == residual (cited 0.000 paired) < central
   references; with the evaluator-call budget shown; NO arm mislabeled.

## Failure criterion
Any deployable arm shown to call the evaluator (mislabeled); a central reference called a deployable
baseline; a headline from < 5 seeds; the residual==anchor citation unsupported.

## Adversarial verification (Ultracode)
A multi-lens Workflow re-derives from the raw per-seed JSON: (1) deployable < central on every seed;
(2) residual == anchor (paired 0.000) is faithfully cited from Q9/Q11; (3) the CI / grouping is correct;
(4) the evaluator-call budget is honest; (5) no over-claim.

## Out of scope
N≥24 (the open frontier — needs a cheaper exact-fault evaluator); the docs close-out (Q13).
