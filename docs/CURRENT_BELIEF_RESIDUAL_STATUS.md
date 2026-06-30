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
| PPO clip (`ppo_clip_actor_loss`) | **ACTIVE_IN_LOSS** (`dynamic_rl.py`, `train_decentralized_rl.py`) | **NOT_PRESENT** | NOT_PRESENT | residual PPO ACTIVE_IN_LOSS (R3) |
| approx_kl / clip_fraction / target_kl | ACTIVE_IN_LOSS | NOT_PRESENT | NOT_PRESENT | logged in residual trainer (R3) |
| CTDE value critic | present (CentralizedGraphCritic) | NOT_PRESENT (scalar moving baseline) | NOT_PRESENT | ACTIVE_IN_LOSS (R3) |
| entropy bonus | CALLABLE (`entropy_coef`) | NOT_PRESENT | NOT_PRESENT | ACTIVE_IN_LOSS (R3, R10) |
| raw-logit L2 / saturation control | NOT_PRESENT | NOT_PRESENT | NOT_PRESENT | ACTIVE_IN_LOSS (R1) |
| CSI-belief auxiliary (`L_CSI`, belief_head) | NOT_PRESENT | NOT_PRESENT | NOT_PRESENT | ACTIVE_IN_LOSS (R2) |
| separate residual head (small-range logits) | NOT_PRESENT (shared ±10 head) | NOT_PRESENT | NOT_PRESENT | ACTIVE (R1) |
| beneficial-edit supervision (repair/safety/edit) | NOT_PRESENT | NOT_PRESENT | NOT_PRESENT | ACTIVE_IN_LOSS (R4–R5) |
| evidence-gated residual action | NOT_PRESENT | NOT_PRESENT (all-edge Bernoulli) | NOT_PRESENT | ACTIVE_IN_DEPLOY (R6) |
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
| R1 | feature standardization + logit-saturation fix (all-frame norm, raw-logit L2, separate small-range residual head, saturation metrics) | NOT_IMPLEMENTED | saturation rate down; recurrent vs memoryless logits no longer bit-identical |
| R2 | CSI belief prediction auxiliary (`belief_head`, `L_CSI`; true CSI = training label only) | NOT_IMPLEMENTED | recurrent belief MSE < memoryless under delay; belief loss in policy loss |
| R3 | residual PPO + CTDE critic (clip/KL/entropy, `A_t=G_t−V`) | NOT_IMPLEMENTED | residual calls `ppo_clip_actor_loss`; KL/clip/EV sane; clamp/collapse reduced |
| R4 | beneficial oracle-edit dataset (ΔC/ΔD/ΔE/ΔL/ΔJ; only positive-gain local edits) | NOT_IMPLEMENTED | non-zero positive-edit rate; no full-oracle-topology imitation |
| R5 | repair/safety/utility/edit heads (supervised) | NOT_IMPLEMENTED | held top-k edit hit rate > random |
| R6 | evidence-gated residual action | NOT_IMPLEMENTED | bad edits gated out; zero-candidate→anchor; budget-safe; 0-eval deploy |
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
**Next: R1** — fix feature standardization (all train frames) + logit saturation (raw-logit L2 + separate
small-range residual head) + saturation metrics, so the temporal signal can pass the actor head.
