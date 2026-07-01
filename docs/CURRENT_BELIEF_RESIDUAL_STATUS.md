# CURRENT_BELIEF_RESIDUAL_STATUS — HEAD vs the Belief-Guided Evidence-Gated Residual PPO target

> ## 🏁 BELIEF-GUIDED EVIDENCE-GATED RESIDUAL PPO — CAMPAIGN COMPLETE (2026-07-01)
> R0–R8 built and tested the full method under the Contract v4 Claim-Path evidence regime, then R10 closed it
> out. **CENTRAL RESULT: no deployable arm beats the `local_hysteresis` anchor on the CURRENT channel (R6/R7)
> OR the STALE channel (R8) at N≤16 — the binding limit is the deployable PRECISION of the beneficial-edit
> direction signal.** The signal genuinely EXISTS (R4, `positive_edit_rate` CI>0) and is locally RANKABLE (R5,
> held top-k CI>0 on random) — the campaign's first deployable-learning positive — but it does NOT CONVERT into
> a deployed feasibility/return gain: the evidence gate is correct/safe/load-bearing yet reproduces the anchor
> at its best operating point on both channels (R6/R8 sweeps: no `tau_edit` beats the anchor), and an adaptive
> anchor-KL trust region also lands at the anchor (R7). The stale-CSI premise is CONFIRMED (stale significantly
> degrades the anchor: urban −0.165, matching Q14 0.80→0.66) but the method does not repair it. Every mechanism
> is verified-correct + adversarially checked (Workflows per stage); each is individually informative. This is
> the **4th independent campaign** (v2-static / D0–D14 dynamic / POMDP-QP-FAR / Belief-Residual) to reach the
> same honest negative — now with the most precise diagnosis (see the 4-chain ledger in §5 and the
> `BELIEF-GUIDED RESIDUAL PPO CAMPAIGN SUMMARY` at the tail of `docs/URBAN_V2X_RESEARCH_LOG.md`).
> **Commits R0–R8 + R10 on `decentralized-marl-trunk`, NOT pushed (owner's decision). Suite 817/0.**

> Authority of record for THIS round (highest priority):
> 1. `docs/MARL-Topology-Belief-Guided-Residual-PPO-TechSpec.md`
> 2. `docs/MARL-Topology-Belief-Guided-Residual-PPO-Workflow.md`
> 3. `docs/MARL-Topology-Development-Contract-v4.md` (Claim-Path evidence) + v3 (still binding)
> Prior campaign: `docs/CURRENT_POMDP_QPFAR_STATUS.md` (Q0–Q13 COMPLETE) + `docs/pomdp_qpfar/Q14_*` addendum.
>
> HEAD at campaign start: `a634928` (Q14 addendum pushed). Stage tags follow Contract v4 §15.

---

## 0. Why this round (the Q14 diagnosis, sharpened)

Q14 produced a PATH-SPECIFIC negative (Contract v4 §5): under stale CSI the deployable anchor loses
feasibility (urban 0.80→0.66 delay-1), and the **REINFORCE residual trainer with a fixed flip-penalty**
does NOT recover it — it is bimodal (3/5 clamp, 2/5 collapse), and cross-frame recurrence is behaviorally
inert because the trained edge logits saturate at the tanh ±10 rail. **Q14 did NOT test residual PPO, a
CTDE critic, a CSI-belief auxiliary, evidence-gated edits, or beneficial-edit supervision — those paths are
NOT_PRESENT in the residual trainer.** This round builds and tests them, one variable per round, under the
Claim-Path evidence regime.

The method (TechSpec): **Belief-Guided Evidence-Gated Residual PPO** — four chains:
1. temporal: stale history → GRU → current-CSI **belief** → edge/residual score (R1–R2)
2. training stability: residual action → old/new logp → PPO clip/KL → CTDE critic (R3)
3. direction supervision: central repair/prune → **beneficial-edit labels** → repair/safety/edit heads (R4–R5)
4. safe exploration: anchor → **evidence-gated** residual candidates → adaptive KL/safety (R6–R7)

---

## 1. Mechanism-Path Matrix (Contract v4 §2) — HEAD snapshot (audited R0, commit-pending)

Status ∈ {NOT_PRESENT, IMPLEMENTED_ONLY, CALLABLE, ACTIVE_IN_LOSS, ACTIVE_IN_EVAL, ACTIVE_IN_DEPLOY}.

| mechanism | static/dynamic graph_mappo trunk | residual_pbrs_train (Q9/Q11) | stale_csi_residual_joint (Q14) | target this round |
|---|---|---|---|---|
| PPO clip (`ppo_clip_actor_loss`) | ACTIVE_IN_LOSS (`dynamic_rl.py`) | **ACTIVE_IN_LOSS (R3 `residual_ppo_train`, per-edge, spy-verified)** | NOT_PRESENT | ✓ done (R3) |
| approx_kl / clip_fraction / target_kl | ACTIVE_IN_LOSS | **ACTIVE_IN_LOSS (R3, target_kl early-stop)** | NOT_PRESENT | ✓ done (R3) |
| CTDE value critic | present (CentralizedGraphCritic) | **ACTIVE_IN_LOSS (R3 `ResidualValueCritic`, EV 0.08–0.56)** | NOT_PRESENT | ✓ done (R3) |
| entropy bonus | CALLABLE (`entropy_coef`) | **ACTIVE_IN_LOSS (R3)** | NOT_PRESENT | ✓ done (R3) |
| raw-logit L2 / saturation control | NOT_PRESENT | NOT_PRESENT | NOT_PRESENT | ACTIVE_IN_LOSS (R1) |
| CSI-belief auxiliary (`L_CSI`, belief_head) | NOT_PRESENT | NOT_PRESENT | NOT_PRESENT | **ACTIVE_IN_LOSS (R2) but NO-OP for recovery** — belief does NOT beat the stale-echo floor (5-seed CI entirely negative, 0/5); leak-free + in-loss verified; `L_CSI` ablatable at R3 |
| separate residual head (small-range logits) | NOT_PRESENT (shared ±10 head) | NOT_PRESENT | NOT_PRESENT | ACTIVE (R1) |
| beneficial-edit supervision (repair/safety/edit) | NOT_PRESENT | NOT_PRESENT | NOT_PRESENT | **ACTIVE_IN_EVAL (R5)** — heads trained (L_edit BCE / L_repair/L_safety Huber) on R4 labels; held top-k beats random (random CI>0, urban mean-positive); local features only |
| evidence-gated residual action | NOT_PRESENT | NOT_PRESENT (all-edge Bernoulli) | NOT_PRESENT | **ACTIVE_IN_DEPLOY (R6)** — gate on frozen R5 heads filters candidate edits (spy); budget-safe, zero→anchor, 0-eval; correct+load-bearing BUT B==anchor (deployed gain negative) |
| adaptive anchor KL / safety constraint | NOT_PRESENT | NOT_PRESENT (fixed flip penalty) | NOT_PRESENT (fixed) | **ACTIVE_IN_LOSS (R7)** — anchor_kl in actor loss + adaptive beta controller (load-bearing, tightens/loosens); residual == anchor (confirmatory negative) |
| D_quorum potential | — | CALLABLE (PBRS) | CALLABLE | edit-label + repair head (R4–R5) |
| stale/partial CSI observation | — | NOT_PRESENT (current CSI) | **ACTIVE_IN_EVAL** (`--csi-mode`) | **ACTIVE in deployed obs (R8)** — delay-1 stale; premise confirmed (drop CI>0) but gate doesn't repair |

**Audit takeaway:** PPO/critic/entropy exist ONLY in the graph_mappo trunk; the residual path is REINFORCE +
moving baseline + fixed flip-penalty, with NO belief/edit-supervision/gating. Contract v4 §7 forbids citing
the trunk's PPO as residual PPO — the residual path must be built and proven on its own.

---

## 2. R-stage gap table (Workflow R0–R10)

| R | deliverable | HEAD status | exit condition |
|---|---|---|---|
| **R0** | freeze Q14 + residual-trainer path audit | **DONE (this commit)** — Mechanism-Path Matrix above; 3 load-bearing audit tests pass (`tests/unit/test_belief_residual_R0_audit.py`): graph_mappo defines PPO/KL; residual source is REINFORCE-only; spy proves a residual update never calls `ppo_clip_actor_loss` (R3 tripwire) | "PPO exists" ≠ "PPO active in residual path" pinned in code ✓ |
| R1 | feature standardization + logit-saturation fix (all-frame norm, raw-logit L2, separate small-range residual head, saturation metrics) | **DONE** — `BeliefResidualActor` (±3 separate head, exposes raw) + `residual_saturation.py`; pilot urban delay-1: old ±10 head frac_logit_near_rail **1.0** + recurrent−memoryless logit delta **0.0 (inert)** → new ±3 head **0.0** rail + delta **0.0257 (passes)**; action_delta still 0 (logit-level only, R3 for topology); 5 load-bearing tests, suite 780/0; Workflow `wfv50jg36` | saturation down ✓ AND recurrent vs memoryless logits no longer bit-identical ✓ |
| R2 | CSI belief prediction auxiliary (`belief_head`, `L_CSI`; true CSI = training label only) | **DONE (REVISE — HONEST NEGATIVE: belief is a no-op CSI predictor)** — `BeliefResidualActor.belief` + `csi_belief.py` + `csi_belief_train.py`. Belief loss ENTERS the policy-actor loss (grads reach belief_head+GRU, spy) and is leak-free (verified). **BUT held belief MSE sits at/above the stale-echo floor** (predict the stale obs) even with leak-free velocity + 80–400 epochs → **does NOT recover current CSI** (learns the echo; correction-direction corr≈−0.035). Recurrence also null (mem−rec CI spans 0). 7 tests (incl. honest-negative pin), suite green; Workflow `wozljm9uf` MAJOR (adopted). | belief-in-loss MET ✓; "recovers CSI / beats stale-echo floor" NOT met → HONEST NEGATIVE (TechSpec chain 1) |
| R3 | residual PPO + CTDE critic (clip/KL/entropy, `A_t=G_t−V`) | **DONE (KEEP — PPO stabilizes; == anchor)** — `residual_ppo_train.py` genuinely calls `ppo_clip_actor_loss` (per-edge ratios, spy), logs approx_kl/clip_fraction, target_kl early-stop; `ResidualValueCritic` EV **0.56** (LayerNorm fix), entropy, raw-L2. PPO **stable** (retention 1.0, 0 collapse) — fixes free-REINFORCE bimodal collapse — BUT residual **== anchor** (edit_rate 0.0): missing-direction-signal (R4–R5), not a PPO failure. 7 tests, suite 794/0; Workflow PENDING | PPO+critic active ✓; clamp/collapse reduced ✓; == anchor → direction-signal gap |
| R4 | beneficial oracle-edit dataset (ΔC/ΔD/ΔE/ΔL/ΔJ; only positive-gain local edits) | **DONE (KEEP — beneficial-edit signal EXISTS)** — `oracle_edit_dataset.py` (teacher-only; anchor + single edit + Δ's, never the oracle topology). 5-seed positive_edit_rate **random 0.096 [0.055,0.137] / urban 0.125 [0.074,0.175]** (CIs strictly >0); repairable 0.26/0.18; safe-prune 0.28/0.75; best ΔJ 0.73/0.12. CENTRAL-reference signal (∝ Q7/Q8) — deployable learning is R5. 6 tests, suite 800/0; Workflow `w1o20fuos` | non-zero positive-edit rate ✓; no full-oracle imitation ✓ → R5 |
| R5 | repair/safety/utility/edit heads (supervised) | **DONE (KEEP — PARTIAL POSITIVE: the R4 signal IS locally learnable)** — `edit_head`/`repair_head`/`safety_head` on `BeliefResidualActor` (local features `[ef, h_u⊙h_v, |h_u−h_v|]` only) + `edit_head_training.py` (L_edit BCE + L_repair/L_safety Huber; held eval). 5-seed held top-k precision−base: **random +0.251 [+0.089, +0.413] (CI>0, 3.2× lift)**, urban +0.184 [−0.005, +0.373] (4.5× lift, spans 0 by 0.005). repair_corr random +0.226 [+0.176,+0.277]. **UNTRAINED control at chance** ([−0.088,+0.094]/[−0.033,+0.093]) → the lift is from LOCAL-feature training, not the metric. 5 tests, suite 805/0; Workflow `w4refc811` | held top-k edit hit rate > random — **random MET decisively, urban met in mean (not 95%-sig at n=5)** → KEEP → R6 |
| R6 | evidence-gated residual action | **DONE (KEEP MECHANISM / HONEST NEGATIVE on deployed gain — deployable CONVERSION gap)** — `evidence_gated_action.py` (`local_candidates` + `evidence_gated_residual`: anchor 0-eval → frozen R5 heads score → repair/safety/edit gate → budget-safe mutual decode; zero-gated→anchor). Mechanism correct+safe+load-bearing (5 tests; budget-safe; 0 unsafe at tau 0.5; 0 eval). **5-seed A/B (tau 0.5): B−A feas spans 0 (random 0.000 [−0.022,+0.022] / urban +0.005 [−0.009,+0.019]) = B==anchor; edit_rate ~0.001 (mean), zero_edit ~0.97; B−C feas +0.095/+0.12 mean (heads suppress; not 95%-sig).** Threshold sweep: every FIRING tau_edit (0.35→0.05) net-negative in mean, unsafe rises urban 0.00→0.09, no tau's B−A lo>0 → **no operating point beats the anchor**. Workflow `wa52tamrf` PASS/PASS/MINOR/MINOR (no MAJOR). suite 810/0 | bad edits gated out ✓ / zero→anchor ✓ / budget-safe ✓ / 0-eval ✓ **MET (mechanism)**; deployed gain **NOT met (B==anchor at best, <anchor when firing)** = R5 ranking's ~40% precision doesn't CONVERT deployably |
| R7 | adaptive anchor KL / safety constraint (replace fixed flip penalty) | **DONE (KEEP MECHANISM / CONFIRMATORY NEGATIVE — == anchor)** — `residual_ppo_train.py` adaptive path: `anchor_kl_penalty` (mean σ(z) over candidates) in the actor loss + `update_beta` (retention<τ→×1.5 / stable+val↑→×0.7) + `_map_retention` (0-eval); one variable vs R3 (adaptive_anchor_kl=False = exact R3). **5-seed A/B: adaptive residual_feas == anchor == fixed EXACTLY (random 0.300 / urban 0.783; adaptive−anchor CI [0.0,0.0]), edit_rate 0.0, retention 1.0, 0/5 diverged; beta_anchor controller load-bearing (tightens AND loosens per seed).** Even with residual_prior=0 + self-loosening, residual == anchor → fixed protection NOT the cause. Workflow `wguwwbury` PASS/PASS/PASS (byte-level repro). suite 815/0 | no collapse ✓ / no clamp-artifact ✓ (retention 1.0, 0 diverged, beta adapts); deployed gain **== anchor** (confirmatory) — closes the "fixed vs adaptive" objection |
| R8 | stale-CSI premise test (does the method repair the stale drop?) | **DONE (HONEST NEGATIVE under the REAL regime; premise CONFIRMED)** — `r8_stale_csi_gen.py` (`build_csi_scenes` with `CsiObservationModel(mode="delay", delay_frames=1)`; heads retrained on STALE features, labels from TRUE evaluator; 3 arms current/stale anchor + stale gated; 0-eval deploy) + `r8_stale_threshold_sweep.py`. **Premise CONFIRMED: stale drop random 0.055 [+0.009,+0.101] / urban 0.165 [+0.118,+0.212] (both CI>0; urban matches Q14 0.80→0.66). NO repair: (gated−stale_anchor) feas random +0.010 [−0.007,+0.027] / urban 0.000 [0,0] (edit 0); sweep — NO tau_edit lo>0, firing net-negative + unsafe rises → no operating point repairs the drop; "room" hypothesis refuted.** 2 tests (leak-free stale overlay); Workflow `w4hpxwg29` PASS/PASS/MINOR (byte-level leak-free repro). suite 817/0 | ≥1 arm repairs the stale drop → **NOT met** (no tau repairs; failing layer localized = deployable direction-signal PRECISION, same as R6, now in the STALE regime) |
| R9 | 5-seed research campaign | NOT_IMPLEMENTED | per-seed + CI + budget + scope |
| R10 | docs close-out | NOT_IMPLEMENTED | what was/ wasn't active; did residual PPO beat anchor; did belief make recurrence useful |

**R1–R5 are preconditions.** Per Workflow §0, no "residual learning failed" claim is permitted until they pass.

---

## 3. Non-downgradable hard constraints (TechSpec §17 + Contract v4)

1. Deployment fully decentralized; deployed actor never reads true current CSI / critic / evaluator / central
   solver / global decoder. 2. Training MAY use true current CSI as a supervised target / critic input, marked
   training-only. 3. True PBFT C/E/L always on the current real channel + closed-form whole-network quorum tail.
   4. Proxy / D_quorum / PBRS / CSI-aux are training signals only, never the final metric. 5. Imitate ONLY
   anchor-positive-gain LOCAL edits, never full central-oracle topology. 6. "PPO" requires actually calling the
   PPO clip loss + logging ratio/approx_kl/clip_fraction/target_kl; "critic" requires critic_parameter_delta /
   value_loss / EV. 7. Do NOT cite the trunk's PPO as residual PPO. 8. Do NOT call recurrence ineffective until
   logit saturation is fixed. 9. Every conclusion needs a Claim Card + Mechanism-Path Matrix row. 10. Failing
   test first; report whether the mechanism changes the final action/topology, not just the forward pass.
   11. held/test never used for teacher / checkpoint / training. Commit per stage to trunk; push = owner's call.

---

## 4. Current stage

**R0 DONE (audit; this commit).** Residual-trainer path audited: PPO/critic/entropy/KL exist only in the
graph_mappo trunk; the residual path is REINFORCE + moving baseline + fixed flip-penalty; no belief / edit
supervision / gating. 3 load-bearing audit tests pin this (incl. the R3 PPO spy tripwire). Q14's negative is
now formally scoped to "REINFORCE-residual + flip-penalty under delay1 urban", NOT residual PPO / belief.
**R1 DONE** (logit-saturation fix). `BeliefResidualActor` (separate ±3 residual head exposing raw) +
`residual_saturation.py` (all-frame standardization / saturation metrics / raw-logit L2 / recurrent-vs-
memoryless delta). Pilot (urban delay-1, after identical logit-pushing): the OLD ±10 shared head reproduces
Q14 (frac_logit_near_rail **1.0**, recurrent−memoryless logit delta **0.0 = inert**); the NEW ±3 head +
raw-L2 is **unsaturated (0.0)** and recurrence now **passes** (delta **0.0257 > 0**). **Honest scope: the fix
is LOGIT-level only — action_delta=0 (the topology is unchanged until a trained policy near decision
boundaries, R3).** 5 load-bearing/effect-on-decision tests; suite 780/0; Workflow `wfv50jg36`.
**R2 DONE — REVISE / HONEST NEGATIVE: the belief auxiliary is a no-op CSI predictor.** `BeliefResidualActor.
belief` + `belief_head` + `csi_belief.py` + `csi_belief_train.py`. The belief loss genuinely ENTERS a training
loss on the policy actor (grads reach `belief_head` AND the shared GRU — spy) and is leak-free (actor input =
stale ef; true current psucc = training-only label) — both verified (Workflow `wozljm9uf` lens 1 PASS). **BUT
the held belief MSE sits AT/ABOVE the trivial stale-echo floor** (predict the stale observed psucc) even with
the new leak-free velocity helper (`leak_free_motion_features`, rel_vel+distance_delta, no leaking `csi_delta`)
and 80–400 epochs: delay2 belief 0.112 vs floor 0.111; 400ep overfits to 0.118; correction-direction
corr≈−0.035. **So the head learns to ECHO the stale input and recovers ~none of the staleness** — a no-op CSI
*predictor* (Workflow lens 2 MAJOR, adopted + reproduced/extended). Recurrence is also null (mem−rec CI spans
0), now secondary. **Disposition: keep the belief machinery (correct + leak-free) but do NOT claim it recovers
CSI; treat `L_CSI` as an ablatable auxiliary at R3.** The "belief-guided" premise (TechSpec chain 1) is
weakened — recoverable value must come from the other chains. 7 tests (incl. `test_belief_does_not_beat_stale_
echo_floor`), suite green. **R3 DONE — KEEP (PPO stabilizes the trainer; residual == anchor = missing direction signal).**
`residual_ppo_train.py` genuinely calls `graph_mappo.ppo_clip_actor_loss` on the residual path (per-edge/
per-agent ratios — `residual_logp_per_edge`; spy-verified, 144 distinct ratios at inner epochs), logs
approx_kl/clip_fraction, target_kl early-stops; `ResidualValueCritic` trains (EV 0.08–0.56 across seeds,
single-run 0.56; LayerNorm + standardized target fix); entropy + R1 raw-L2 active; `L_CSI` ablatable.
**5-seed A/B: PPO 0/5 collapse (random + urban, retention 1.0) vs free-REINFORCE 1/5 each — PPO eliminates
the collapse.** BUT PPO residual == anchor (edit_rate 0.0 all seeds): no beneficial deviation → a
missing-direction-signal outcome (R4–R5), NOT a PPO failure. 7 tests, suite 794/0; Workflow `wj8pzo538`
4-lens **PASS**. The campaign's remaining lever is now the DIRECTION signal (beneficial-edit supervision).
**R4 DONE — KEEP (the campaign's first genuinely hopeful result: a beneficial-edit DIRECTION signal EXISTS).**
`oracle_edit_dataset.py` enumerates local add/remove/swap over the anchor and scores each with the TRUE
evaluator → ΔC/ΔD_quorum/ΔE/ΔL/ΔJ (ΔJ in the dense reward). 5-seed positive_edit_rate **random 0.096
[0.055,0.137] / urban 0.125 [0.074,0.175]** — both CIs strictly above 0; anchor-failure repairable 0.26/0.18;
safe-prune 0.28/0.75; best ΔJ 0.73 (random). So the anchor is NOT a local optimum — contrary to the prior
"nothing beats the anchor" pattern. **Honest scope: this is a CENTRAL-reference (training-only) signal
(consistent with Q7 22%-repair / Q8 47%-prune); it does NOT yet show a DEPLOYABLE head can learn it from local
features (R5) or beats the anchor (R6/R8).** Teacher-only (held never passed); stores anchor + single edit
(never the oracle topology). 6 tests, suite 800/0; Workflow `w1o20fuos`.
**R5 DONE — KEEP (PARTIAL POSITIVE: the campaign's first decisive deployable-LEARNING positive — local
features CAN predict the R4 beneficial edits).** `edit_head`/`repair_head`/`safety_head` appended to
`BeliefResidualActor` (per-edge LOCAL features `[ef, h_u⊙h_v, |h_u−h_v|]` only — NO evaluator / true CSI; the
R4 evaluator scores are training-only LABELS) + `edit_head_training.py` (`build_edit_examples`,
`train_edit_heads` = L_edit BCE + L_repair/L_safety Huber, SEPARATE held eval, `topk_metrics`). **5-seed held
top-k edit precision − random base rate: random +0.251 [+0.089, +0.413] (CI strictly >0, 3.17× lift), urban
+0.184 [−0.005, +0.373] (4.50× lift, CI spans 0 by 0.005 — mean strongly positive but NOT 95%-significant at
n=5).** repair_corr random +0.226 [+0.176, +0.277] (CI>0). **The UNTRAINED head's precision−base sits at
chance on both regimes ([−0.088,+0.094]/[−0.033,+0.093]) → the lift is produced by training the heads on LOCAL
features, NOT a metric artifact.** This answers the deployable-LEARNING question POSITIVELY (decisive on random,
mean-positive on urban) — it is NOT a learning gap; R4 proved the signal exists, R5 proves local features can
learn to RANK it. **Honest scope: R5 proves the RANKING is locally learnable, NOT yet that gating the residual
action on the heads improves the DEPLOYED topology (R6), and urban is not 95%-significant at n=5.** 5 tests,
suite 805/0; Workflow `w4refc811`.
**R6 DONE — KEEP MECHANISM / HONEST NEGATIVE on the deployed gain (a deployable CONVERSION gap).**
`src/marl_topology/training/evidence_gated_action.py` (`local_candidates` deployable enumeration +
`evidence_gated_residual`): compute the anchor (0 eval) → enumerate LOCAL candidates → score with the FROZEN R5
heads (`edit_scores`, local features only) → gate (add iff repair_pred≥τ_r ∧ σ(edit_logit)≥τ_e; remove iff
safety_pred≤τ_s ∧ σ(edit_logit)≥τ_e) → apply via the budget-safe mutual decode; zero gated → anchor exactly.
The heads move ACTIVE_IN_EVAL (R5) → ACTIVE_IN_DEPLOY (R6). Mechanism is CORRECT + SAFE + load-bearing (5
failing-first tests: gate calls heads in the decision path / zero→anchor / bad-head→anchor & good-head→edits /
budget-safe / no-evaluator; 0 unsafe at tau 0.5; 0 eval at decision — all independently re-verified by Workflow
`wa52tamrf`). **BUT the deployed gain is NEGATIVE: 5-seed A/B (tau 0.5) B−A feasibility spans 0 (random 0.000
[−0.022,+0.022], urban +0.005 [−0.009,+0.019]) = B == anchor (the calibrated heads suppress edits, edit_rate
~0.001 mean, zero_edit ~0.97); the threshold sweep is downhill from the empty gate — every FIRING tau_edit
(0.35→0.05) is net-negative in mean with unsafe_edit rising (urban 0.00→0.09), and NO tau's B−A lower bound
> 0 → no operating point beats the anchor.** B > random-gate C in mean (+0.095/+0.12, not 95%-sig): the trained
heads' only deployable value is SAFE SUPPRESSION. **Diagnosis (Contract v4 §13): a DATA/precision limit — the
R4 signal exists (R4) and the R5 local ranking is real (~40% top-k), but that precision does NOT CONVERT into a
net deployed gain (a harmful edit's feasibility cost exceeds a beneficial edit's gain) — NOT a mechanism/path/
trainer/leak bug.** This SHARPENS the campaign's binding limit from "no direction signal" (pre-R4) to "the
direction signal's deployable-rankable PRECISION is insufficient to beat the anchor at N≤16", and empirically
pre-empts R7's adaptive-anchor-KL premise (optimal anchor-deviation = 0). Chain 3 fully characterized: EXISTS
(R4) → RANKABLE (R5) → NOT deployably CONVERTIBLE (R6). Workflow `wa52tamrf` 4-lens+synthesis MINOR (PASS/PASS/
MINOR/MINOR, no MAJOR; 3 wording fixes applied). suite 810/0.
**R7 DONE — KEEP MECHANISM / CONFIRMATORY NEGATIVE (adaptive anchor-KL residual == anchor).** `residual_ppo_
train.py` adaptive path (additive; `adaptive_anchor_kl=False` = exact R3): `anchor_kl_penalty` (mean σ(z) over
candidate edges = deviation from the flip-nothing anchor) in the actor loss + `update_beta` adaptive controller
(retention<τ → ×1.5 tighten / retention≥τ ∧ val↑ → ×0.7 loosen / clamp) + `_map_retention` (0-eval signal). One
variable vs R3 = the anchor pull, FIXED → ADAPTIVE. **5-seed A/B (deployed MAP decode, true PBFT feas): adaptive
residual_feas == anchor_feas == fixed_resid_feas EXACTLY (random 0.300 / urban 0.783; adaptive−anchor CI
[0.0,0.0], adaptive−fixed CI [0.0,0.0]); edit_rate 0.0; retention 1.0; 0/5 diverged. The beta_anchor controller
is LOAD-BEARING (per-seed trajectories tighten AND loosen, 22-23 up / 30-31 down steps across 5 seeds, distinct
per seed, never diverges) — yet the residual never leaves the anchor.** Even with the fixed pull removed
(residual_prior=0) and a self-loosening trust region, the PPO advantage gives no beneficial-direction gradient
(R3 finding; the direction lives in the frozen R5 heads, not this policy). **CONFIRMATORY NEGATIVE — CLOSES the
"did you try adaptive, not fixed?" objection: the fixed anchor protection was NOT why R3/R6 == anchor.** THREE
independent mechanisms (R3 fixed-prior PPO, R6 evidence gate, R7 adaptive KL) all land at the anchor → R6's
precision-limit binding stands. Workflow `wguwwbury` 3-lens+synthesis **PASS/PASS/PASS** (independent byte-level
repro: beta trajectory byte-matched, anchor_kl in loss, R3 reproduced exactly flag-off, leak-free; no doc edits
required). suite 815/0.
**R8 DONE — HONEST NEGATIVE under the REAL regime (the campaign's premise test); premise CONFIRMED.**
`r8_stale_csi_gen.py` turns the stale-CSI overlay ON (`CsiObservationModel(mode="delay", delay_frames=1)` →
`build_csi_scenes`): the deployed actor observes STALE CSI (cols 0–3 = frame t−1, +[age, mask]) while the
evaluator stays on the TRUE current channel (leak-free — test: CSI cols diverge at t≥1, evaluator byte-identical
stale-vs-current). Heads RETRAINED on stale features (labels from the TRUE evaluator, training-only); evidence-
gated action deployed under stale CSI, 0 eval. Arms: current_anchor (ceiling) / stale_anchor (degraded) /
stale_gated. **Premise CONFIRMED: the stale overlay significantly degrades the anchor — stale_drop random 0.055
[+0.009,+0.101] / urban 0.165 [+0.118,+0.212] (both CI>0; urban 0.815→0.650 matches Q14 0.80→0.66). But the
method does NOT repair it: (gated−stale_anchor) feas random +0.010 [−0.007,+0.027] spans 0 / urban 0.000 [0,0]
(edit 0 → == stale_anchor); the stale-CSI threshold sweep — NO tau_edit ∈ [0.05,0.5] has a (gated−stale_anchor)
feas CI lo>0, firing edits net-negative in mean with rising unsafe (urban 0.00→0.045), same downhill pattern as
R6's current channel. The stale-degraded anchor's "room" does NOT rescue the method — the "room" hypothesis is
REFUTED.** (The 1-seed pilot's +0.0625 was noise.) Diagnosis: a DATA/precision limit in the STALE regime — the
deployable direction-signal precision holds the method to the anchor on BOTH the current (R6/R7) AND stale (R8)
channels; NOT a leak/mechanism/trainer bug. 2 tests; Workflow `w4hpxwg29` 3-lens+synthesis MINOR (PASS/PASS/
MINOR; byte-level leak-free repro; 2 wording fixes applied — inferred-not-measured stale precision, overlay
active-in-deployed-observation). suite 817/0. **CAMPAIGN COMPLETE (experimental): no deployable arm beats the
`local_hysteresis` anchor on the current OR the stale channel at N≤16.**
**R10 DONE — honest close-out (pure docs).** COMPLETE banner (this doc, top) + `BELIEF-GUIDED RESIDUAL PPO
CAMPAIGN SUMMARY` at the tail of `docs/URBAN_V2X_RESEARCH_LOG.md` + Belief-Residual banner on
`docs/CURRENT_HEAD_STATUS.md` + `AGENTS.md` pointer; the 4-chain diagnosis (§5 below). Suite unchanged 817/0.

---

## 5. FINAL 4-CHAIN DIAGNOSIS (R10 close-out) — where the method's value is, and where it stops

The TechSpec method had four chains. Each was built, made load-bearing, and adversarially verified; the
campaign's value is the PRECISE localization of the binding limit.

| chain | mechanism | status | result (path-specific) |
|---|---|---|---|
| 1. temporal / belief | GRU over stale history → current-CSI belief → edge score | **✗ non-load-bearing** | R1 fixed logit saturation but the ACTION is unchanged (logit-level only); R2 the CSI belief is a NO-OP — it does not beat the stale-echo floor (5-seed CI entirely negative). Recurrence null. |
| 2. trainer stability | residual → old/new logp → PPO clip/KL → CTDE critic; adaptive anchor-KL | **✓ correct, but no direction** | R3 residual PPO GENUINELY wired (per-edge ratio, target_kl, CTDE EV 0.08–0.56) — ELIMINATES the free-REINFORCE collapse (0/5 vs 1/5) but residual == anchor (edit 0). R7 adaptive anchor-KL (load-bearing controller) ALSO == anchor. The trainer is not the limit. |
| 3. direction supervision | central beneficial-edit labels → repair/safety/edit heads | **EXISTS → RANKABLE → NOT CONVERTIBLE** | R4 the signal EXISTS (`positive_edit_rate` random 0.096 / urban 0.125, CI>0; anchor NOT a local optimum). R5 LOCAL heads can RANK it (held top-k − base random +0.251 CI>0, 3.2× lift; urban mean-positive) — the campaign's FIRST deployable-learning positive. R6/R8 it does NOT CONVERT to a deployed gain (no `tau_edit` beats the anchor on current OR stale). |
| 4. evidence-gated action | gate residual on the repair/safety heads; adaptive KL | **✓ correct/safe, == anchor** | R6/R8 the gate is budget-safe, zero→anchor, 0-eval, load-bearing — but at its best operating point it reproduces the anchor (suppresses edits); firing edits is net-negative (harmful-edit feasibility cost > beneficial-edit gain). |

**BINDING LIMIT = the deployable PRECISION of the beneficial-edit direction signal**, on BOTH the current
channel (R6/R7) and the stale channel (R8, the actual failure regime — premise confirmed, no repair). The
direction signal is real and locally learnable, but a deployable local policy cannot convert its ~40%-precision
ranking into a net feasibility/return gain over the near-optimal (or stale-degraded) `local_hysteresis` anchor
at N≤16 — a harmful edit costs more feasibility than a beneficial edit gains, and the heads cannot separate them
sharply enough to fire only the winners.

**This is the 4th independent campaign** (v2-static / D0–D14 dynamic / POMDP-QP-FAR / Belief-Residual) to reach
the same honest negative — now localized to the direction-signal precision, NOT its existence (R4), its
learnability (R5), the trainer (R3/R7), the gate (R6/R8), the temporal/belief chain (R1/R2), or any leak.

**Open frontier (deferred):** N≥24 with a cheaper exact-fault evaluator; a higher-precision direction signal
(e.g. richer local features or a learned edge encoder that lifts held top-k precision well above ~40%) is the
only lever that could plausibly flip the deployed conversion — the campaign shows precision, not mechanism, is
the wall.
