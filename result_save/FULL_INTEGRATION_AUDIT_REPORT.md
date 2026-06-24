# MARL-Topology — Full Integration Audit + Instrumented Re-Run (Honest Report)

**Date:** 2026-06-24 · **Branch:** decentralized-marl-trunk · **Audit base commit:** `073b590` (audit follow-up commits `3044050`, `062237c`, this report) · **Driver of audit:** owner challenge to the v2 campaign's "simple baseline beats every sophisticated mechanism" conclusion · **Reference:** `docs/MARL-Topology-Full-Integration-Audit-Checklist.md`

---

## 0. Executive summary (the one-line correction)

The v2 campaign's central result is **valid but mis-scoped**. The honest statement is:

> On the **corrected static single-step (T=1) N≤16 contextual-bandit subtask**, the sophisticated mechanisms that were *genuinely activated in a headline* (Graph-MAPPO central critic, per-agent PPO ratio, BCSP action, COMA counterfactual credit, PNA actor) **did not beat the simple Graph-MAPPO + MLP baseline**.

It does **NOT** support the stronger claim *"the full dynamic CTDE-SCQ-MARL architecture failed"*, because:

1. **The dynamic task was never trained.** The formal training entry is a T=1 bandit; `two_timescale_env.py` is imported only by its unit test. The dataset itself has **no multi-step trajectories** (all 60 sequences are length-1, `time_step=1`).
2. **Chance / CVaR / Pareto were never activated in any headline** (`chance=false`, `pareto_archive=false` in every run; CVaR is a primitive never wired into a loss).
3. **SCQ was never in a headline** (diagnostic, single seed only).
4. **Recurrence and ω-preference-conditioning structurally degenerate at T=1** (the GRU never carries hidden state across episode steps; ω is hardwired neutral and never swept).

The owner's three suspicions were therefore **correct**: the dynamic environment is an opt-in primitive not wired into training/evaluator; single-step testing cannot validate the components whose value is inherently temporal; and "dynamic data infrastructure ≠ dynamic RL task."

This report (a) audits every checklist section with file/artifact evidence, independently adversarially verified; (b) checks production-config conformance; (c) records a **fully-instrumented re-run** (R7 vs 8b, 3 seeds, op_corrected) with the complete artifact set the checklist demands; (d) runs the Temporal Value Test on the real env code; and (e) states the one decision that is the owner's to make.

---

## 1. Checklist §13 conclusion template

```
实验名称：       v2 campaign headline arms (Phase 8 R7-vs-8b; Phase 11 MLP-vs-PNA) + instrumented re-run
git commit：     073b590 (campaign) ; audit instrumentation 3044050 ; this re-run on 3044050+062237c
数据版本：       op_corrected, environment_math_version = v2-corrected-2026-06-23
                 (PhysicsRegime: fault_model=fixed_set, one_hop_relay=True, relay_hops=3,
                  timeout_aware_latency=True, tau=0.9, node_count_choices=(8,12,16))
训练入口：       scripts/train/train_decentralized_rl.py  (SINGLE-STEP T=1 contextual bandit)
命令：           python scripts/train/train_decentralized_rl.py --baseline graph-mappo [--counterfactual]
                   --cold-start --updates 80 --k-cf 4 --shards <op_corrected 3001-3016> --seed {0,1,2}
seed：           0,1,2 (re-run) ; 0-4 (original headline)
split：          split_seed=7, held_frac=0.4, val_scenes=40 ; fit/val/held disjoint (asserted)
是否 headline eligible：  YES for the STATIC T=1 N≤16 subtask only ; NO for any dynamic/temporal claim

环境接入：
[PASS]    safe quorum (n>=3f+1, 2q-n>f, q<=n-f) — enforced at load
[PASS]    fixed fault set (fault_model=fixed_set, consistent across phases)
[PASS]    one-hop relay (one_hop_relay=True, relay_hops=3 ; legacy double-count not used)
[PARTIAL] phase accounting — phase-specific PBFT message plan (R4) is a PRIMITIVE, NOT wired into the
          production evaluator (vectorized_objective_stack_evaluator still uses the all-pairs matrix)
[PASS]    timeout-aware latency (E[min(T,B)]; failed topologies pay the phase budget)
[PASS]    tri-state (finite-search miss=unknown, never certified_infeasible ; unknown trains by default ;
          fit/val/held witness isolation asserted)

模型接入：
[PASS]    BCSP (sample_decentralized_bcsp_action ; unordered ; no permutations on production path)
[PASS]    local mutual decoder (local_mutual_assemble unique for rollout+eval ; argsort ablation-only)
[PASS]    per-agent PPO ratio (element-wise over (scene,agent) ; not a joint sum)
[PASS]    graph critic (V) — grad-on, EV~0.99, critic_parameter_delta>0, excluded from deployed artifact
[PASS]    counterfactual Q — critic_sees_action=True, detached one-hot, EV~0.5-0.95, eval/scene=1
[WIRED, INACTIVE] SCQ — wired & correct but never in a headline (diagnostic single-seed only)
[NOT ACTIVE] chance / CVaR / Pareto — none activated in any headline (chance/pareto off ; CVaR primitive-only)
[PASS (arch) / FAIL (temporal+pref)] PNA — actor wired+active 5 seeds ; recurrence degenerate at T=1 ;
          ω fixed neutral, never swept
[FAIL]    dynamic task — NOT wired (T=1 bandit ; two_timescale_env unwired ; dataset has no trajectories)

结论：
只能下静态子任务结论。复杂机制（已激活的：critic / per-agent ratio / BCSP / COMA / PNA）在
修复后的静态单步 N≤16 contextual-bandit 上没有超过简单 Graph-MAPPO+MLP 基线。
动态任务 / chance / CVaR / Pareto / SCQ / recurrence / preference 从未进入 headline，结论不覆盖它们。
```

---

## 2. Detailed audit (8 sections, independently adversarially verified)

Each section was audited by a dedicated agent reading the real code+artifacts, then a second agent adversarially tried to refute every "wired"/"activated" claim. Status uses the checklist ladder.

### 2.1 Environment math — **PASS** (one PARTIAL)
- **Corrected evaluator** `ACTIVE_IN_RESEARCH`: the *same* `Stage21ObjectiveStackEvaluator` is used for rollout reward, held eval, and (when on) SCQ counterfactuals. Headline + re-run shards are all `op_corrected/_op_shard_3001-3016`. The env-math is stamped *in the shard's generator config* (`PhysicsRegime`: `fault_model='fixed_set'`, `one_hop_relay=True`, `relay_hops=3`, `timeout_aware_latency=True`) — not hardcoded.
- **Safe quorum / fixed fault set / one-hop relay / timeout latency / tri-state**: all `ACTIVE_IN_RESEARCH`, wired into the trunk + active in headline. Tri-state: finite-search miss = `unknown` (never `certified_infeasible`); `unknown` scenes train by default; fit/val/held isolation asserted at runtime (`assert_split_isolation`).
- **GAP — phase-specific PBFT accounting (R4) `IMPLEMENTED_ONLY`:** `pbft_message_plan.py` (pre_prepare: primary→backups; prepare/commit: validator votes) is defined and unit-tested but **not referenced by the production evaluator** — the training evaluator still uses the legacy all-pairs message matrix. Honest mark: verified primitive, not integrated.

### 2.2 Dynamic task — **FAIL (not wired)**
- `two_timescale_env.py` (`TwoTimescaleTopologyEnv` with persistence, reconfiguration cost, γ) exists but is imported **only** by `tests/unit/test_two_timescale_env.py`.
- The trunk is T=1: docstring L30-32 ("Temporal stays OFF, single-step contextual-bandit"), `reward_of` L176-178 ("single-step (T=1) bandit with no states/transitions"), L651 (`single-step A = r - V (GAE at T=1)`). No episode loop, no γ discounting, no topology persistence, no reconfiguration cost, no return-over-horizon.
- **The dataset has no trajectories**: all 60 `(scene, sequence_id)` groups have a single `time_step=1` frame.
- The Temporal Value Test (Δ_H) had only ever run on synthetic toy frames (`_const_cost`/`_alternating_cost`) — see §5.

### 2.3 Actor + action path — **PASS** (static T=1 scope)
- BCSP sampler `sample_decentralized_bcsp_action` (trunk L608) is the production path; `itertools.permutations` exists only in the unused `_ordered_topk_entropy` (legacy PL), not reachable from the graph-mappo arm. b≥m → independent-Bernoulli fast path. Entropy normalized per agent's legal action count.
- Per-agent PPO ratio: `logp_old_flat`/`adv_flat` flattened over `(scene, agent)`; `ppo_clip_actor_loss` ratio is element-wise (not a joint sum). Epoch-0 ratio = 1 (logits unchanged since rollout).
- `local_mutual_assemble` is the **unique** decoder for rollout *and* held eval; `global_argsort_assemble` is ablation-only.

### 2.4 Critic — **PASS** (static T=1 scope)
- V critic: forward grad-ON in the update (not `no_grad`), `opt_c.step`, actor params isolated (separate optimizer). **Measured `critic_parameter_delta` 7.2-7.5 > 0**; explained variance **~0.99**; checkpointed+resumable; **excluded from the deployed actor artifact** (stored separately as a training artifact).
- Q critic (8b): `critic_sees_action=True`, input = realized active-edge one-hot (detached), Q changes with action, trains toward reward (EV 0.5-0.95), V↔Q resume arch-mismatch guard present.

### 2.5 Counterfactual PPO — **PASS** (static T=1 scope)
- Each agent has its own `A_i = Q(s,S) − E_{S̃_i~π_i}[Q(s,S̃_i,S_{-i})]`; within-scene variance > 0 (unit-tested); `S̃_i` sampled from `θ=logits[incident]/temp` (independent of the realized `S_i`); `S_{-i}` fixed; counterfactual re-decoded through the mutual decoder; **counterfactual Q forwards consume no evaluator calls — `evaluator_calls_per_scene == 1` in all 5 original + 3 re-run 8b seeds** (COMA is budget-neutral). The actor loss consumes the per-agent advantage (`adv_flat` from `per_agent_adv`), not a scene scalar.

### 2.6 SCQ — **WIRED_BUT_INACTIVE**
- Correctly wired: `--scq` requires `--counterfactual`; exact ΔR_i from the real `reward_of`; counterfactual targets computed **once per update** (not per critic epoch); SCQ loss enters the **critic** objective only (never actor gradient); `scq_critic_difference_error` logged.
- **Never in a headline** — only the diagnostic `_q_scq_corrected` (single seed). The difference error did **not** improve (≈2.2× worse over training) and held Q-fidelity did not improve (ρ 0.244→0.071, prior measurement). Honest mark: **verified primitive, opt-in, not promoted**.

### 2.7 Chance / CVaR / Pareto — **NONE activated in any headline**
- **Chance**: wired into the reward (`-lam_chance·1[c<τ]`, sign-flexible dual) but `chance=false` in **every** headline → `lam_chance` stayed 0, penalty never applied.
- **CVaR**: `cvar_shortfall` is a unit-tested **primitive with zero coupling** to the training loop / checkpoint selection. Must not be reported as "CVaR trained".
- **Pareto**: `--pareto-archive` wired into final checkpoint selection but `pareto_archive=false` in every headline. Even if enabled, it is **degenerate at T=1**: `hypervolume` is hardcoded `0.0` and entries differ only by reliability_violation/energy/latency at fixed neutral ω.

### 2.8 PNA / recurrent + preference — **PASS (arch) / FAIL (temporal + preference)**
- PNA actor wired + active in Phase 11 (5 seeds, op_corrected), using BCSP + the mutual decoder. → licenses only "PNA on static T=1 N≤16 without ω-sweep does not beat MLP" (it was in fact *worse*: MLP 0.594 vs PNA 0.503).
- **Recurrence degenerate at T=1**: the shared GRUCell runs across K message-passing rounds *within one forward*, never across episode steps → mathematically a larger static net.
- **Preference fixed**: ω hardwired neutral `(0.5,0.5)` in `_neutral_omega`; `forward_logits` never passes ω; no `--omega` arg; never swept in train or eval. → cannot claim a preference-conditioned Pareto policy was tested.

---

## 3. Production-config conformance (checklist point 5)

| Mechanism | Production config requirement | Conformance |
|---|---|---|
| **Dynamic task** | episode_length>1, hold_interval, γ, reconfig cost | **NON-CONFORMING** — no dynamic config exists; T=1 by design |
| **Graph-MAPPO** | per-agent ratio, central graph critic, eval_calls/scene=1 (ARMS registry) | **CONFORMS** — clip=0.2, ppo_epochs=4, target_kl=0.01, eval/scene=1.0 (measured) |
| **Counterfactual PPO** | per-agent COMA, action-conditioned Q, budget-neutral | **CONFORMS** — k_cf=4, critic_sees_action=True, eval/scene=1.0 (measured, all seeds) |
| **SCQ** | exact ΔR from real evaluator, critic-only, extra evaluator calls logged | **CONFORMS as a mechanism, but OFF in production default** (only diagnostic) |

Production default (`CURRENT_HEAD_STATUS`: `--baseline graph-mappo --actor mlp`) = the simple baseline, which is exactly what is validated. The fixed `ACTION_DISTRIBUTION_VERSION` constant was stale (`ordered_plackett_luce...`); corrected to `bcsp_v1_unordered_subset` in this audit.

---

## 4. Instrumented re-run (checklist point 6 — the full recording)

**Driver:** `result_save/_audit_rerun/_driver.sh` · **Command per run:** `python scripts/train/train_decentralized_rl.py --baseline graph-mappo [--counterfactual] --cold-start --updates 80 --k-cf 4 --shards <op_corrected 3001-3016> --seed {0,1,2} --out-dir <dir>` · **stdout/stderr** captured per run. Every run emitted the full artifact set: `data_manifest.json`, `split_manifest.json`, `mechanism_activation.json`, `seed_manifest.json`, `critic_metrics.json`, `rl_result.json`, `_rl_artifacts.pt`, `stdout.txt`, `stderr.txt`.

### (1) Data provenance (`data_manifest.json`)
- 16 shards `op_corrected/_op_shard_3001..3016.pkl`, each with SHA256 (e.g. 3001 = `af6025466a5dd70e…`). `environment_math_version = v2-corrected-2026-06-23`. Generator config (incl. `env_math_regime`) recorded. Splits: per-`split_manifest.json` fit/val/held with scenario IDs, node-count distribution {8,12,16}, solvability distribution, **trajectory_length distribution = {"1": N} for all splits** (the honest T=1 marker).

### (2) Mechanism activation (`mechanism_activation.json`)
- `dynamic_task.enabled = false` (episode_length 1, reconfiguration_cost_nonzero false).
- R7: `critic.graph_mappo=true`, `counterfactual.enabled=false`, `action.distribution=bcsp`, per-agent ratio true.
- 8b: `counterfactual.enabled=true`, `k_cf=4`, `critic.critic_sees_action=true`.
- All runs: `scq.enabled=false`, `constraint.chance_enabled=false`, `constraint.cvar_in_loss=false`, `pareto.enabled=false`, `actor.preference_conditioned=false`.

### (3) Training-log evidence (final-update `critic_metrics.json`)
| run | critic_param_delta | explained_var | eval_calls/scene | actor_gnorm | critic_gnorm | subset_card | active_edges |
|---|---|---|---|---|---|---|---|
| r7 s0 | 7.475 | 0.998 | 1.0 | 0.136 | 0.27 | 2.27 | 11.73 |
| r7 s1 | 7.443 | 0.985 | 1.0 | 0.430 | 0.55 | 1.93 | 9.66 |
| r7 s2 | 7.221 | 0.995 | 1.0 | 0.029 | 0.44 | 1.91 | 9.79 |
| 8b s0 | 8.449 | 0.502 | 1.0 | 0.018 | 19.53 | 3.48 | 8.75 |
| 8b s1 | 7.892 | 0.945 | 1.0 | 0.009 | 2.09 | 3.47 | 11.75 |
| 8b s2 | 8.384 | 0.746 | 1.0 | 0.054 | 2.75 | 3.73 | 12.59 |

All `critic_parameter_delta > 0` (the critic genuinely trains); `eval_calls/scene = 1.0` for both arms (COMA budget-neutral, fair A/B). Also logged: actor/critic loss, entropy, per-agent KL, clip fraction, dual trajectory (`history`).

### (4) Dynamic-task instrumentation
Not applicable — `dynamic_task.enabled=false`. Recorded honestly as such (no episode length / hold interval / PBFT-rounds-per-macro-step / reconfiguration / state-transition traces exist because the loop is T=1). The Temporal Value Test stands in for this (§5).

### (5) Results
| arm | per-seed raw (held) | mean | 95% CI |
|---|---|---|---|
| **R7** (graph-mappo, MLP, shared advantage) | 0.594, 0.641, 0.641 | **0.625** | [0.558, 0.692] |
| **8b** (+ COMA counterfactual credit) | 0.594, 0.656, 0.609 | **0.620** | [0.539, 0.701] |
| **paired 8b − R7** | 0.000, +0.016, −0.031 | **−0.005** | [−0.065, +0.054] |

The paired difference spans 0 → **8b COMA does not beat the R7 baseline** on the static subtask. This **reproduces the original headline** (+0.009, CI [−0.033, +0.052]) with full provenance. Checkpoint selection rule: keep-best on VAL raw (never held). Baseline = same evaluator, same op_corrected, same cold-start. Each run ≈ 19-28 min wall-clock (CPU).

### (6) Model artifacts
Per run: `_rl_artifacts.pt` (deployed actor state + normalization; **critic stored separately, never in the deployed path** — D1), plus `critic_parameter_delta` recorded. (Optimizer/Pareto/resume checkpoints are emitted when `--ckpt-every`/`--pareto-archive` are set; default off here.)

---

## 5. Temporal Value Test (checklist §3.3 — on the real env code)

`scripts/diagnostics/temporal_value_test_real.py` → `result_save/temporal_value_test_real.json`.

- **Part A (mechanism, real code):** running the actual `TwoTimescaleTopologyEnv` / `temporal_value_test` over a controlled 2-frame case and sweeping reconfiguration cost `r` and hold interval `H` reproduces **Δ_H = max(0, 2r − H) exactly on all 20 cells**. Temporal modeling matters (Δ_H>0) only in 4/20 cells — **only when `2·reconfig_cost > hold_interval`**, i.e. ≈0 at realistic PBFT hold intervals (H≥2) unless reconfiguration is very expensive. This validates the R5 deferral on the real code path.
- **Part B (real data):** the corrected evaluator's energy/feasibility for each scene's named candidate topologies (`topology_variants`) is recorded on held scenes (e.g. an N=8 scene: only `greedy_reliability_raw` feasible, c=1.0, E=0.064; `full_graph_raw` infeasible). **All sequences are length-1** → at a single channel realization the per-frame optimum is fixed → a faithful Δ_H is **identically 0 by construction**. A nonzero temporal value can only be measured on multi-frame / multi-realization data that op_corrected does not contain.

**Conclusion:** the static bandit is the correct model **for the data that exists**; any dynamic-task conclusion requires new trajectory / channel-re-realization data.

---

## 6. Honest summary (checklist point 7)

**What is genuinely established (and now fully instrumented + reproduced):** On the corrected static T=1 N≤16 contextual-bandit subtask, the activated CTDE machinery — central Graph-MAPPO critic (EV~0.99, Δθ>0), per-agent PPO ratio, BCSP unordered-subset action, local mutual decoder (train==deploy), and budget-neutral COMA counterfactual credit — is correct and wired, and **does not beat the simple Graph-MAPPO + MLP baseline** (paired 8b−R7 = −0.005, CI [−0.065,+0.054]). The deployed path is fully decentralized (critic excluded from the artifact).

**What was NOT tested (so no conclusion is licensed):**
- The **dynamic T>1 task** (unwired; no trajectory data).
- **Chance / CVaR / Pareto** reliability mechanisms (never on in any headline).
- **SCQ** at headline scale (diagnostic single-seed only; no Q-fidelity gain even there).
- **Recurrent temporal state** and **ω-preference-conditioned Pareto policy** (degenerate at T=1 / never swept).
- **Phase-specific PBFT accounting** (primitive, not in the production evaluator).
- **Large-N (N≥24)** generalization (separate open frontier).

**Corrected campaign headline:** replace *"the full architecture's sophisticated mechanisms don't help"* with *"on the corrected static single-step small-scale subtask, the activated mechanisms don't beat the simple baseline; the dynamic task, reliability constraints, SCQ-at-scale, temporal recurrence, and preference-conditioning remain untested."*

### The one decision that is the owner's to make
Testing the **genuine dynamic task** requires a substantial build that **reverses the documented R5 deferral**:
1. Wire `two_timescale_env` into the trunk as a real T>1 MDP (episode rollout, topology persistence, reconfiguration cost, γ-return, recurrent hidden-state carry, per-frame ω).
2. Generate **multi-frame / multi-realization channel data** per scene (the current dataset has none).
3. Re-measure with the dynamic-task instrumentation already prepared.

The Temporal Value Test (§5) is the bounded evidence on whether this is worth doing: at realistic H_PBFT, Δ_H≈0, suggesting limited temporal value unless reconfiguration cost is high. **I will not start this build without explicit owner approval.**

---

## 7. Reproduction

```bash
# Audit instrumentation tests
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/unit/test_run_instrumentation.py -q

# Instrumented re-run (R7 + 8b, 3 seeds) — emits the full artifact set under result_save/_audit_rerun/
bash result_save/_audit_rerun/_driver.sh

# Temporal Value Test (real env code + real corrected costs)
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python scripts/diagnostics/temporal_value_test_real.py
```

Artifacts: `result_save/_audit_rerun/{r7,8b}_seed{0,1,2}/` (data_manifest / split_manifest / mechanism_activation / seed_manifest / critic_metrics / rl_result / stdout / stderr / _rl_artifacts) ; `result_save/temporal_value_test_real.json`.
