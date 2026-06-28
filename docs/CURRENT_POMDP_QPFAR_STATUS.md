# CURRENT_POMDP_QPFAR_STATUS — HEAD vs the POMDP-QP-FAR specs

> Authority of record for THIS round (highest priority):
> 1. `docs/MARL-Topology-POMDP-QP-FAR-Technical-Spec.md` (technical design)
> 2. `docs/MARL-Topology-POMDP-QP-FAR-Workflow.md` (Q0–Q13 engineering workflow)
> Governing rigor contract (unchanged, binding): `docs/MARL-Topology-Development-Contract-v3.md`.
> Prior campaign close-out: `docs/CURRENT_DYNAMIC_REPAIR_STATUS.md` (D0–D14, all 10 gaps closed).
>
> This file is the frozen diff between HEAD (`edc0340`, D0–D14 dynamic-repair complete, clean at
> origin) and the two POMDP-QP-FAR specs. It is updated at each Q-stage decision. Status tags follow
> Contract v3 §1.

---

## 0. The two-axis problem this round attacks (Spec §1)

The D0–D14 campaign produced an **honest negative**: every sophisticated mechanism (recurrence,
velocity, COMA, SCQ, chance/CVaR, Pareto, PNA) is verified-correct but gives **no statistically
significant headline gain**, and the learned arms sit **below** the deployable local heuristics,
which sit at/below the central myopic reference. The binding limit was localized to two coupled
causes, NOT to "MARL fails":

- **Axis A — temporal degeneracy.** Under full current-CSI + previous-topology observation the task
  is ~Markov (constant-velocity geometry is perfectly predictable), so memory has little to recover
  (D8/D13: recurrence Δfeas CI spans 0; Temporal Value Δ_H>0 in only ~21% of scenes).
- **Axis B — feasibility-region plateau.** Reliability `c = C(x)` is a whole-network multi-phase
  quorum-tail conjunction; deep in the infeasible region `C(x)≈0` AND a single local edit barely
  moves it (`C(x⊕e)−C(x)≈0`). The reward is already **dense in c** (`r=(c−τ)−…`), so the hardness is
  GEOMETRY (a flat sub-feasible plateau with a distant cliff), not a binary reward.

POMDP-QP-FAR targets both: **stale/partial CSI** makes history valuable (Axis A); a **quorum-deficit
potential `D_quorum`** fills the plateau and **feasible-anchored residual learning** starts the
policy next to the deployable manifold instead of searching the full BCSP space (Axis B).

---

## 1. What HEAD already has (reusable substrate) — code facts

| Capability | Where (verified) | Status | Reuse for POMDP-QP-FAR |
|---|---|---|---|
| Two-timescale dynamic env (T>1, previous topology, reconfig cost) | `training/two_timescale_env.py`; `training/dynamic_frames.py::DynamicScene` | ACTIVE_IN_RESEARCH | the MDP `s_t` PBRS/Q9 needs |
| Per-frame actor observation (nf/ef/ei, prev topo, step) | `dynamic_frames.py::DynamicScene.observation` (L152) | ACTIVE_IN_RESEARCH | **Q1 injection point** — `ef` col 0 = current psucc |
| Motion features (D5), incl. an explicit **`csi_age=0.0` placeholder "for stale-CSI"** | `dynamic_frames.py::_motion_edge_block` (L194–216) | WIRED_BUT_INACTIVE (opt-in) | **Q1** turns the placeholder into a real age |
| Reward dense in c: `r=(c−τ)−λ_b·g_b−β·1[feas]·E/E_ref`, `×hold_interval` | `scripts/train/train_decentralized_rl.py::reward_of` (L166–198); `training/dynamic_rl.py` (L126–132) | ACTIVE_IN_RESEARCH | reward/eval read `obs["context"]` (true channel), **not** `ef` → Q1 split is clean |
| Whole-network closed-form quorum tail (Poisson-binomial DP, capped tail bucket) | `protocol/quorum_tail.py::heterogeneous_quorum_tail` (L33) | ACTIVE_IN_RESEARCH | **Q3** reuses the SAME DP but keeps the full pmf P(S=k) |
| Phase-specific PBFT accounting (pre-prepare/prepare/commit) | `protocol/pbft_accounting.py`, `protocol/pbft_message_plan.py` | ACTIVE (opt-in, D4) | **Q3** per-phase/per-receiver shortfall structure |
| Deployable local heuristics: `local_threshold_action`, **`local_hysteresis_action`** (local-only, 0 evaluator calls) | `training/dynamic_baselines.py` (L34–64) | ACTIVE_IN_RESEARCH (D7) | **Q5 anchor** to imitate; **Q6** residual base |
| Central myopic reference (grouped separately, evaluator-greedy) | `training/dynamic_baselines.py::evaluate_central_reference` | ACTIVE (reference, NOT baseline) | grouping discipline for Q12 |
| Real 4-RSU urban data (`--dyn-data urban`) + random ablation | `dynamic_frames.py::sample_dynamic_urban_scenes`; `dynamic_rl.py` dispatch | ACTIVE (D1) | Q12 data axis |
| Dynamic mechanisms, all opt-in/default-off/verified: COMA `--counterfactual`, SCQ `--scq`, chance/CVaR/Pareto `--chance`/`--pareto-archive`, PNA `--dynamic-actor-arch pna` | `training/counterfactual_credit.py`, `scq_supervision.py`, `dynamic_reliability.py`, `models/dynamic_pna_actor.py` | each VALIDATED_NEGATIVE / NEGATIVE_BUT_SCOPE_LIMITED (D9–D13) | Q11 re-tests PNA inside residual frame |
| D13 campaign driver + frozen result | `scripts/diagnostics/dynamic_d13_campaign.py`; `result_save/dynamic_d13_campaign.json` (30.5 KB) | FROZEN baseline (Q0) | the negative control everything is measured against |

**Note on the current `−τ`:** it is a constant offset on the reward, **not** Ng–Harada potential-based
shaping (AGENTS.md is explicit). Genuine PBRS `F_t=γΦ(s_{t+1})−Φ(s_t)` is only legitimate in the T>1
dynamic task (Q9) and requires a terminal-zero potential — this is a NEW capability, not present at HEAD.

---

## 2. POMDP-QP-FAR gap table (Q0–Q13) — what is missing vs the specs

| Q | Deliverable (spec) | Planned artifact | Current status |
|---|---|---|---|
| Q0 | Freeze D13 negative as control baseline | `result_save/dynamic_d13_campaign.json`, `docs/dynamic_repair/D13/decision.md` | **artifacts present on disk** — formalize freeze this round → `docs/pomdp_qpfar/Q0/` |
| Q1 | Stale/partial CSI observation model (`current\|delay\|partial\|delay_partial`, age, mask) | `src/marl_topology/training/csi_observation_model.py` | **DONE** (commit pending) — opt-in `--csi-mode`; leak-free (evaluator on true channel); default byte-identical; 9 unit + suite 715/0; delay+partial smokes exit 0; adversarial PASS |
| Q2 | CSI-prediction + Temporal-Value health check under stale CSI | `scripts/diagnostics/csi_prediction_health.py` | **DONE** (commit pending) — GATE PASS: under delay/partial, temporal beats identity+memoryless on 3/3 seeds × 3/3 modes (delay1 0.104→0.080); current identity=0 (trivial); 5 unit + suite 720/0; adversarial PASS (no leak, causal GRU) |
| Q3 | Quorum shortfall `D_quorum` diagnostics (per-phase/per-receiver, mean/max/CVaR, worst) | `src/marl_topology/protocol/quorum_deficit.py` | **DONE** (commit pending) — exact Poisson-binomial `E[(q−S)_+]`; tail bucket == reliability quorum tail (~1e-19); demonstrates gradient where C is flat (p=0.5→0.9: C≈0, D 2.74→1.39); 10 unit + suite 730/0; NOT in reward (gated on Q4); adversarial PASS |
| Q4 | `D_quorum` ↔ true `C` alignment test (Spearman ΔD/ΔC, top-k repair hit, ΔD↑ C↓ rate) | `scripts/diagnostics/quorum_deficit_alignment.py` + `training/quorum_deficit_bridge.py` + evaluator `_reliability_inputs` refactor | **DONE** (commit pending) — GATE PASS (safe+conditional): 0% false-improvement all seeds/data; cond-Spearman 0.69–0.95 where C moves; urban overall 0.75–0.80; flat_C 0.85–0.99 confirms plateau; D_quorum AUTHORIZED for Q9 PBRS; 5 unit + suite 735/0; adversarial PASS |
| Q5 | `local_hysteresis` imitation actor (BCSP subset NLL, decoded F1, held feas, switches) | `local_hysteresis_proposals` + `_hysteresis_teacher_trajectory` + `scripts/diagnostics/anchor_imitation.py` | **DONE (mixed, honest)** — BC-imitation reproduces anchor on random (F1 0.89) but FAILS on urban (NLL diverges, F1 0.24; diagnosed: budget-64 RSU vs budget-2 vehicle conflicting shared-edge targets). → Q6 uses anchor as DIRECTLY-COMPUTED base (`x=H⊕Δ`, zero residual=anchor EXACT), not BC-imitation. 4 unit + suite 739/0 |
| Q6 | Residual action space (zero residual = anchor; add/remove/swap; local mutual decoder) | `training/residual_action.py::residual_decode` | **DONE** (commit pending) — anchor base + add/remove/swap/full; zero-residual=anchor EXACT all modes; decentralized (per-node, no global sort), budget-safe, 0-eval; 6 unit + suite 745/0 + real-shard smoke (random+urban); adversarial PASS zero issues |
| Q7 | Add-only repair from anchor-failure scenes, guided by `D_quorum` | `training/residual_repair.py` + `scripts/diagnostics/add_only_repair.py` | **DONE (partial+)** — central greedy D_quorum repair fully fixes 22% of random anchor-failures (vs 0% naive max-add → guidance matters), +0.36 C, 100% retention; urban 0 (deeply-infeasible); deployable head deferred to Q9; 4 unit + suite 749/0 |
| Q8 | Conservative prune from feasible anchor (remove low-risk edges, keep feasibility) | `training/residual_repair.py` (prune+safety) + `scripts/diagnostics/conservative_prune.py` | **DONE (partial+)** — SAFE: 100% feasibility retention + 0 critical deletions (both); urban energy down on 47% feasible anchors (+slight C gain), random no benefit (relay overhead, sparse minimal); deployable safety head deferred Q9; 5 unit + suite 754/0 |
| Q9 | Full residual + PBRS (`Φ=−D_quorum`, terminal Φ=0, eval without shaping) | `training/potential_shaping.py` + residual sampler in `residual_action.py` | **PART 1 DONE** — PBRS primitive telescopes + preserves small-MDP optimum (Ng-Harada, 3 potentials); D_quorum potential telescopes on real episodes; residual sampler/logp RL-ready; 6 unit + suite 760/0. **PART 2 DONE** — residual+PBRS REINFORCE trainer (eval-no-shaping); HONEST NEGATIVE: matches anchor (no improvement); urban UNSTABLE without anchor trust-region (Q5 recurs, retention→0), `--anchor-reg 0.5` fixes it (retention 1.0); 9 unit / suite 763/0 |
| Q10 | Local edge handshake / correlated sampling (endpoint score exchange, shared score) | decoder extension | NOT_IMPLEMENTED (current mutual activation is independent) |
| Q11 | Re-test PNA inside the residual frame | campaign arm | NOT_IMPLEMENTED |
| Q12 | Multi-seed × multi-CSI-mode × multi-N campaign | `scripts/diagnostics/` new driver | NOT_IMPLEMENTED |
| Q13 | Docs / report / README / research-log close-out | docs | NOT_IMPLEMENTED |

**Ordering invariant (Spec §workflow):** Q1–Q4 are preconditions. No full RL training (Q7+) before
Q1 (stale CSI verified observation-only) AND Q4 (`D_quorum` alignment) PASS. One variable per round;
failing test first; per-stage `experiment_plan.md` + `decision.md`.

---

## 3. Non-downgradable hard constraints carried into this round (Spec §16 + Contract v3)

1. Deployment fully decentralized; deployed actor reads ONLY stale/partial observed CSI `ĝ_t`, age,
   motion features, previous topology, public protocol params — **never** true current CSI, critic,
   global decoder, solver, searcher, or evaluator.
2. True PBFT reliability / energy / latency ALWAYS computed on the **true current channel** `g_t` and
   the closed-form whole-network quorum tail. Stale/partial CSI changes **only the actor observation**.
3. `D_quorum`, quorum deficit, potential shaping are **training-only proxies** — never replace the
   final metric, and `D_quorum` must PASS the Q4 alignment test before entering any reward.
4. PBRS only in the T>1 dynamic task, with terminal potential = 0; final evaluation uses no shaping.
5. `local_hysteresis` is a **deployable anchor**, not a central teacher; residual policy must report
   anchor **retention** + repair success + mutual acceptance, not just mean return.
6. Central reference / teacher / deployable baseline reported in **separate groups**.
7. Only **one** temporal perturbation at a time (stale CSI first; hidden blockage / random intent /
   battery decay deferred and never co-activated with stale CSI).
8. Every mechanism needs a runtime `mechanism_activation` artifact; default-off ≠ tested; smoke/pilot
   params never headline; ≥5 seeds + CI for any conclusion; conclusions scoped to activated mechanisms.
9. Commit per stage to `decentralized-marl-trunk`; **do NOT push** (owner's decision).

---

## 4. Current stage

**Q0 DONE** (commit 05765d6) — D13 control frozen + re-derived. **Q1 DONE** — stale/partial CSI
observation model (`csi_observation_model.py`), opt-in `--csi-mode {current,delay,partial,delay_partial}`,
leak-free + byte-identical default, 9 unit / suite 715/0, delay+partial smokes exit 0, adversarial PASS.
**Q2 DONE** — CSI-prediction health check: GATE PASS (recurrence beats identity+memoryless 3/3 seeds ×
3/3 stale modes; recurrent actor justified Q5+). **Q3 DONE** — `quorum_deficit.py`: exact Poisson-binomial
`D_quorum`, bit-identical to the reliability's quorum tail, gradient where C flat; NOT in reward.
**Q4 DONE** — D_quorum↔C alignment GATE PASS (safe+conditional): 0% false-improvement on all seeds/data,
conditional Spearman 0.69–0.95 where C moves, urban overall 0.75–0.80, flat_C 0.85–0.99 confirms the
plateau empirically; D_quorum AUTHORIZED for Q9 PBRS (Ng-optimum-preserving regardless; Q9 measures if it
helps); suite 735/0; adversarial PASS.

**🔓 Q1–Q4 PRECONDITIONS ALL MET → residual RL track unblocked.** **Q5 DONE (mixed, honest)** — BC-imitation
of the deployable anchor reproduces it on random (F1 0.89) but FAILS on urban (NLL diverges, F1 0.24;
diagnosed: budget-64 RSU vs budget-2 vehicle → conflicting shared-edge BC targets; decoder budget-top-k ≠
threshold rule). This MOTIVATES the residual-base design over a BC-imitation foundation. **Q6 DONE** — `residual_action.py`
residual decode (anchor base + add/remove/swap/full; zero-residual=anchor EXACT; decentralized per-node no
global sort; budget-safe; 0-eval); 6 unit / suite 745/0 / real-shard smoke (random+urban) / adversarial
PASS zero issues. **Q7 DONE (partial+)** — `residual_repair.py` central greedy D_quorum-guided add repair on anchor-failure
frames: fully fixes 22% of random failures (vs 0% naive max-add → guidance matters; adding the RIGHT edges
beats the MOST), +0.36 true-C improvement, 100% anchor retention; urban 0 (3 deeply-infeasible NLOS
failures); central reference (~96 evals/repair); deployable learned head DEFERRED to Q9. **Q8 DONE (partial+)** —
conservative prune SAFE everywhere (100% feasibility retention + 0 critical-edge deletions); urban lowers
energy on 47% of feasible anchors (+slight C gain via interference reduction), random no cost benefit
(relay overhead, sparse anchors minimal); central reference, deployable safety head deferred Q9. **Q9 PART 1 DONE** — PBRS
mechanism: `potential_shaping.py` (Φ=−D_quorum, telescopes to −λΦ_0, terminal Φ=0) PROVEN optimum-preserving
on a small MDP (Ng-Harada, 3 potentials) + telescopes exactly on real episodes (random/urban); residual
RL sampler/logp (`sample_residual`/`residual_logp`/`residual_decode_from_flips`) makes the Q6 action space
PPO-trainable (sampler consistent, budget-safe, zero-flips=anchor). 6 unit / suite 760/0 + real-shard PBRS
smoke. **Q9 PART 2 DONE** — residual+PBRS REINFORCE trainer (`residual_pbrs_train.py`, eval-no-shaping).
HONEST NEGATIVE: residual+PBRS policy MATCHES the anchor (no improvement, retention 1.0); urban RL UNSTABLE
without an anchor trust-region (Q5 instability recurs — retention→0, feasibility 0.65→0.04), `--anchor-reg
0.5` fixes it (retention 1.0). PBRS correct/optimum-preserving but doesn't beat the strong deployable anchor
(central ceilings Q7-22%/Q8-47% + campaign "learned≈deployable" pattern). 9 unit / suite 763/0.
**🏁 Q9 COMPLETE.** **Next: Q10** — local edge handshake (two-round endpoint score exchange + shared edge
score, reduce mutual-acceptance mismatch, control-comm cost recorded, deployment-decentralized). See
`docs/pomdp_qpfar/Q*/` for per-stage artifacts.
