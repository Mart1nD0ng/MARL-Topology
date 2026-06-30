# R4 — decision (beneficial oracle-edit dataset) — KEEP: a beneficial-edit signal EXISTS → R5

**Result: KEEP (a positive, and the campaign's first genuinely hopeful one).** Contrary to the prior
expectation that the `local_hysteresis` anchor is near a local optimum, the oracle-edit dataset shows a
**non-trivial fraction of single LOCAL edits over the anchor have a positive true-objective gain (ΔJ>0,
safe)** — so a beneficial-edit DIRECTION signal (the lever R3 showed the trainer lacks) **exists** at N≤16.
This is a CENTRAL-reference capability (consistent with Q7 22%-repair / Q8 47%-prune); it does NOT yet show a
DEPLOYABLE policy can learn it from local features (R5) or beat the anchor at deployment (R6/R8). Verification:
Workflow PENDING; 5-seed CI: `oracle_edit_metrics.json` (PENDING).

## What changed (a teacher dataset; never a deployment mechanism)
`src/marl_topology/training/oracle_edit_dataset.py`: per frame, over the anchor `x_H`, enumerate LOCAL edits
(add = budget-feasible non-anchor edges; remove = anchor edges; swap = bounded top-k×top-k); score each with
the TRAINING evaluator (true current channel + closed-form quorum tail) → ΔC, ΔD_quorum, ΔE, ΔL, and
**ΔJ = reward_of(x_H⊕e) − reward_of(x_H)** (the same dense reward the RL uses). `supervised_records` keeps
ONLY `{ΔJ>margin ∧ safe}`. Stores anchor + single edit descriptor + Δ's — **never the full oracle topology**.
Teacher-only (train scenes; held never passed). A content hash is logged.

## Effect-on-Decision / metrics
PILOT (1 seed, 4 scenes, 16 frames):
| metric | random | urban |
|---|---|---|
| positive_edit_rate (margin 0) | **0.085** (42/494) | **0.224** (110/490) |
| positive_edit_rate (margin 0.01) | 0.055 (27) | 0.124 (61) |
| best_edit_gain_max (ΔJ) | **0.714** | 0.101 |
| anchor_failure_repairable_rate | **0.333** (12 infeasible frames) | n/a (0 infeasible) |
| safe_prune_rate | n/a (4 feasible) | **0.75** (12/16 feasible) |
| evaluator_calls | 1036 | 1028 |

So on random, 1/3 of anchor-failures are repairable by a single add/swap (best ΔJ 0.71); on urban, 22% of
edits are positive and 75% of feasible frames admit a safe energy-cutting prune.

**5-seed CI (`oracle_edit_metrics.json`) — both positive_edit_rate CIs entirely above 0:**
| metric | random | urban |
|---|---|---|
| positive_edit_rate | **0.096 [0.055, 0.137]** | **0.125 [0.074, 0.175]** |
| anchor_failure_repairable_rate | 0.255 | 0.177 |
| safe_prune_rate | 0.276 | 0.746 |
| best_edit_gain_max (ΔJ) | 0.726 | 0.120 |

The beneficial-edit signal is robust across seeds (CIs strictly positive).

## Path-specific scope (Contract v4 §5)
> "Scored by the TRUE evaluator, the local-edit space over the `local_hysteresis` anchor contains a non-trivial
> fraction of positive-gain safe edits (random ~8.5% / urban ~22% at margin 0; repairable failures + safe
> prunes). The anchor is NOT a local optimum. This is a CENTRAL-reference (training-only) signal — it does NOT
> establish that a DEPLOYABLE head can predict these edits from local features (R5) or that the deployed
> policy beats the anchor (R6/R8)."

## Decision rule (R4 exit)
positive_edit_rate is clearly > 0 (not ~0) AND anchor_failure_repairable_rate (random) + safe_prune_rate
(urban) are > 0 → **KEEP → R5** (train repair/safety/utility/edit heads on the positive edits; held top-k edit
hit rate must beat random). The honest open question for R5: can local features predict these gains? If R5's
heads cannot beat random on held, THAT is where the deployable route fails (a learning gap, not a no-signal gap).

## Tests (failing-first → pass)
6 in `tests/unit/test_belief_residual_R4_oracle_edit.py` (fail on HEAD: module absent → pass). Full suite
**800/0**.

## Acceptance (Contract v4 §15)
- Claim Cards ✓ / Mechanism-Path Matrix ✓ / load-bearing tests pass ✓ (6/6, suite 800/0) / activation
  artifact ✓ / pilot + 5-seed raw ✓ / decision ✓ / scope explicit ✓.
- Exit condition: beneficial edits exist (non-zero positive_edit_rate + repairable/prune) → KEEP.
- Decision: **KEEP** — proceed to R5.
- Next: **R5** — repair/safety/utility/edit heads, supervised on the R4 positive edits; held top-k edit hit
  rate > random; safe-remove precision; repair recall. If supervision fails on held → STOP before PPO (the
  deployable learning gap).

## Adversarial verification (Workflow `w1o20fuos`, 4 lenses) — overall PASS
- **ΔJ + DELTAS CORRECT — PASS.** dJ = the SAME dense `reward_of` the RL uses (lam_c=lam_b=1.0, beta=0.1,
  mode dense) on the true current channel; ΔC/ΔE/ΔL via `topology_reliability`, ΔD via `topology_quorum_
  deficit` (true evaluator). Independent recompute of 54 records → **0 mismatches** (incl. non-zero examples).
- **SIGNAL REAL — PASS.** All 10 dataset hashes reproduced byte-for-byte; per-seed CIs recomputed (t.975(df=4)
  =2.776): random 0.096 [0.055,0.137], urban 0.125 [0.074,0.175] — **both strictly above 0**. Top edits are
  genuine **infeasible→feasible repairs** (anchor C=0.0 → edited C=1.0 above τ, energy cut); best-gain CI
  random **[0.573, 0.878]**. Not an artifact.
- **TEACHER-ONLY, NO LEAK, NO FULL ORACLE — PASS.** No held/val/test param; record keys carry no
  `edited_topology`/`oracle`; `anchor` is the deployable base; `edge` is a single str or 2-tuple; the deployed
  residual decode makes **0 evaluator calls**; `dataset_hash` deterministic.
- **HONEST CENTRAL-REFERENCE SCOPE — PASS.** Framed as a training-only signal (∝ Q7/Q8); does NOT claim
  deployable learning works (R5) or beats the anchor (R6/R8); the open question (can local features predict
  the gains?) is correctly deferred to R5.

**Verdict: KEEP — the beneficial-edit signal exists and is verified; proceed to R5 (deployable learning).**
