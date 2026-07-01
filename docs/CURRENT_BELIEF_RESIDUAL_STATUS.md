# CURRENT_BELIEF_RESIDUAL_STATUS — HEAD vs the Belief-Guided Evidence-Gated Residual PPO target

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
| adaptive anchor KL / safety constraint | NOT_PRESENT | NOT_PRESENT (fixed flip penalty) | NOT_PRESENT (fixed) | ACTIVE_IN_LOSS (R7) |
| D_quorum potential | — | CALLABLE (PBRS) | CALLABLE | edit-label + repair head (R4–R5) |
| stale/partial CSI observation | — | NOT_PRESENT (current CSI) | **ACTIVE_IN_EVAL** (`--csi-mode`) | belief input (R2) |

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
| R7 | adaptive anchor KL / safety constraint (replace fixed flip penalty) | NOT_IMPLEMENTED | no retention=0 collapse and no edit_rate=0 clamp |
| R8 | full-method pilot (6 arms × urban delay1/current, random delay1) | NOT_IMPLEMENTED | ≥1 learned arm beats anchor or repairs the stale drop, OR the failing layer is localized |
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
MINOR/MINOR, no MAJOR; 3 wording fixes applied). suite 810/0. **Next: R7** (adaptive anchor-KL/safety — a
confirmatory check, expected == anchor per the sweep) → R8 pilot → R10 honest close-out; OR proceed to the
close-out given the decisive sweep.
