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
| Q4 | `D_quorum` ↔ true `C` alignment test (Spearman ΔD/ΔC, top-k repair hit, ΔD↑ C↓ rate) | `scripts/diagnostics/quorum_deficit_alignment.py` | NOT_IMPLEMENTED (**gates Q7/Q9 reward use**) |
| Q5 | `local_hysteresis` imitation actor (BCSP subset NLL, decoded F1, held feas, switches) | teacher + BC path in trunk | NOT_IMPLEMENTED (anchor heuristic exists; imitation does not) |
| Q6 | Residual action space (zero residual = anchor; add/remove/swap; local mutual decoder) | residual head in actor + trunk | NOT_IMPLEMENTED |
| Q7 | Add-only repair from anchor-failure scenes, guided by `D_quorum` | trunk arm | NOT_IMPLEMENTED |
| Q8 | Conservative prune from feasible anchor (remove low-risk edges, keep feasibility) | trunk arm | NOT_IMPLEMENTED |
| Q9 | Full residual + PBRS (`Φ=−D_quorum`, terminal Φ=0, eval without shaping) | trunk reward | NOT_IMPLEMENTED (requires Q4 PASS) |
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
`D_quorum`, bit-identical to the reliability's quorum tail, demonstrates the gradient where C is flat
(p=0.5→0.9: C≈0 but D 2.74→1.39); 10 unit / suite 730/0; NOT in reward; adversarial PASS. **Next: Q4** —
D_quorum ↔ true C alignment test (Spearman ΔD/ΔC, top-k repair hit, ΔD↑-C↓ rate under local edits on real
topologies; builds the topology→matrices bridge). **D_quorum may enter reward (Q9) ONLY if Q4 passes.**
See `docs/pomdp_qpfar/Q*/` for per-stage artifacts.
