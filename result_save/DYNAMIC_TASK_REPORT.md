# Dynamic (Two-Timescale, T>1) MARL — Build + Honest Test Report

**Date:** 2026-06-25 · **Branch:** decentralized-marl-trunk · **Scope:** owner-authorized build that wires `two_timescale_env` into the trunk, generates genuine multi-frame channel data, and tests the full (recurrent) model. This **reverses the documented R5 deferral** (the v2 campaign had kept the task at a static T=1 bandit). Every claim here is measured on real code/data; negatives are stated plainly.

> **STATUS:** §1–§5, §7 (data/infra/temporal-test/architecture/instrumentation/manifest) are final. §6 (headline results) and §8 (verdict) are filled from the D4 run + adversarial verification.

---

## 1. Executive summary

The static v2 campaign deferred temporal modeling (R5: at realistic PBFT hold intervals the toy Temporal Value Test gave Δ_H≈0). The owner authorized building the genuine dynamic task. This report:

1. **Builds genuine multi-frame channel data** (§2): moving-vehicle trajectories on the production operating-point urban v2x_37885 channel — vehicles move (`advance_scene`) so links evolve **predictably**, the temporal signal a recurrent actor can exploit. The node set + all-pairs candidate edge-id action space are invariant across frames, so the reconfiguration cost `|E_t △ E_{t-1}|` is well defined.
2. **Re-runs the Temporal Value Test on the REAL data** (§3): with the correct candidate set, temporal value (Δ_H>0) appears in **~21% of scenes**, concentrated in the upper quartile, only at non-trivial reconfiguration cost; the **median scene has zero temporal value**. With full per-frame CSI the task is **~Markov in (current channel, previous topology)** — a strong, pre-registered prediction that a memoryless *history-aware* actor should suffice and pure cross-frame recurrence should add little.
3. **Wires the dynamic T>1 episode rollout into the trunk** (§4): a fully-gated `--dynamic` arm (T=1 path byte-identical) with an episode-recurrent actor (per-node hidden carried across frames), recurrent PPO, a per-frame critic, and the deployed decoder — two arms (recurrent vs memoryless) differing ONLY in cross-frame memory.
4. **Tests the full model honestly** (§6) with the complete instrumentation set the owner asked for.

**Central result (D4):** _[FILLED FROM D4 — recurrent vs memoryless paired CI + the myopic-greedy reference]._

---

## 2. Multi-frame channel data (§D1, commit `d872b81`)

**What a "frame" is.** A trajectory is one scenario evolved over `T` macro-frames: vehicles move at constant velocity (15–30 m/s, highway/arterial V2X), the RSU stays fixed (`advance_scene`, pure/immutable). Each frame's channel is the **real Stage-21 measurement on the moved geometry** — link success probability / latency / energy from `context.link_records`, which the actor observes through the edge features. The node set is fixed (vehicles move, never appear/disappear) and candidate edges are all node pairs, so the **edge-id action space is identical across frames** → the reconfiguration cost (symmetric edge difference) is well defined.

**Verified properties** (`tests/unit/test_dynamic_frames.py`, 5 tests + probes):
- node/edge-id sets invariant across frames; frame 0 reproduces the static scene exactly (the T=1 limit);
- the channel evolves with motion (per-frame mean link-success-prob oscillates; SA-measured per-frame feasibility flips, e.g. `[1,1,0,0,1,1,1,1]`);
- the observation encodes the current channel + previous topology (a per-edge flag + per-node previous degree) + step index, via `graph_payload`'s existing hooks.

**Why mobility (not i.i.d. fading).** Independent shadowing realizations vary the channel but are **unpredictable** → a temporal actor cannot anticipate them (myopic-optimal). Mobility gives **predictable** link evolution — the only substrate where cross-frame recurrence can add value. (Sweep: at the operating-point regime with 15–30 m/s mobility, 9/12 scenes show a relevant sparse-topology consensus change across the window.)

**Data manifest** (`result_save/dynamic_data_manifest.json`): deterministic by seed (train `seed*1000+1`, held `seed*1000+777`), `environment_math_version = v2-corrected-2026-06-23`, full generator config, per-split scenario IDs, node-count distribution (mixed {8,12,16}), trajectory-length distribution (all = T frames), solvability-family distribution (≈balanced feasible_sparse/infeasible/near_threshold), and a SHA256 content hash of the frame geometries for byte-verifiable regeneration.

---

## 3. Temporal Value Test on the real data (§D2, commits `1f703c4` → `b152bdd`)

`scripts/diagnostics/dynamic_temporal_value.py` composes the per-frame trunk objective (cost = −`reward_of` dense) into a `two_timescale_env` episode and computes Δ_H = J_myopic − J_horizon via the exact DP over the canonical candidate set (the 6 named topologies: empty / single_best / sparse_quorum / greedy_reliability / full_graph / random), swept over reconfiguration cost `e_edge`.

| e_edge | Δ_H mean | Δ_H median (p50) | frac scenes Δ_H>0 |
|---|---|---|---|
| 0.0 | 0.000 | 0 | 0% |
| 0.03 | 0.052 | 0 | 21% |
| 0.1 | 0.198 | 0 | 21% |
| 0.3 | 0.748 | 0 | 21% |

**Honest finding:** temporal value (Δ_H>0) appears in **~21% of scenes (≈5/24)**, concentrated in the upper quartile, only at non-trivial reconfiguration cost; the **median scene is horizon-optimal under the myopic per-frame policy.** Because the actor observes the full current channel each frame, the task is **~Markov in (current channel, previous topology)** — both in the observation. **Pre-registered prediction for §6:** a memoryless history-aware actor should suffice; pure cross-frame recurrence should add little. (A prior measurement reported ~8%; that used a degenerate candidate set — `topology_variants` is a dict keyed by name, and the buggy code iterated the string keys. Fixed in `b152bdd`. The myopic-greedy reference's measured held per-frame feasibility is **~0.22** — see §6.1.)

---

## 4. Architecture: the dynamic episode wired into the trunk (§D3, commit `279115e` + opt.)

**The MDP (Spec S3.3–3.6).** At each frame the actor observes (current channel, previous topology, step), samples a topology (BCSP per-agent → `local_mutual_assemble`), holds it for `hold_interval` PBFT micro-rounds, and receives `reward = base_objective − reconfig_cost`, `reconfig_cost = (e_edge+l_edge)·|E_t △ E_{t-1}|`. The return is the γ-discounted sum to the episode end; the advantage is `A_t = G_t − V(s_t)`.

**Episode-recurrent actor** (`models/dynamic_recurrent_actor.py`): per-node hidden state carried **across frames** (the cross-frame recurrence the static actors lack — their GRU runs only across message-passing rounds). LayerNorm + soft-bounded (scaled-tanh) logits keep the recurrent BPTT from blowing the BCSP sampler up. The **two arms differ ONLY in whether the hidden carries** (recurrent) or resets each frame (memoryless) — a controlled cross-frame-memory ablation on identical architecture/capacity. 8 tests (carried hidden changes output, BPTT reaches the GRU, symmetric/equivariant, no-NaN).

**Recurrent PPO + per-frame critic** (`training/dynamic_rl.py`): rollout (no-grad, hidden carried) → discounted returns → recurrent re-roll with BPTT to recompute per-(frame,agent) PPO logp → per-agent clipped ratio (Spec S9.2); the per-frame centralized critic regresses `V(s_t) → G_t`. 4 mechanics tests (return recursion, reward = base − reconfig, BPTT reaches the GRU, eval metrics in range).

**Decentralization (D1) + train==deploy.** The rollout decoder IS the deployed torch-free `local_mutual_assemble`; the actor uses local features + physical-neighbour messages only (no global state / decoder); the critic is training-only. The `--dynamic` branch returns before any T=1 code → the static path is byte-identical (verified: T=1 graph-mappo smoke unchanged).

---

## 5. Instrumentation (the full recording the owner asked for)

Every `--dynamic` run emits, under its `--out-dir`:
- **`mechanism_activation.json`** — `dynamic_task.enabled=true`, `episode_length`, `hold_interval`, `gamma`, `reconfig_e_edge/l_edge`, `reconfiguration_cost_nonzero`, `mobility_speed_mps`, `dt_s`; actor `model_id` + `cross_frame_recurrence` + arm; critic enabled + per-frame; action distribution = bcsp, decoder = local_mutual_assemble, per-agent ratio.
- **`training_history.json`** — per update: actor loss, critic loss, critic explained variance, actor/critic grad norms, **critic parameter delta**, per-agent KL, clip fraction, entropy, mean subset cardinality, **mean switches per frame**, and the periodic val per-frame feasibility / episode return.
- **`held_traces.json`** — per held scene: per-frame topology size, switches, feasibility, base reward, reconfiguration cost (the **state-transition trace**).
- **`dynamic_result.json`** — held per-frame feasibility, mean episode return, mean switches/frame, best-val return, n_train/n_held, config.
- The headline driver additionally captures **stdout.txt / stderr.txt** per run.

---

## 6. Results — full-model dynamic test (§D4)

**Driver:** `scripts/diagnostics/dynamic_headline.py` (subprocess per run; stdout/stderr captured). **Command/config:** §7. 3 seeds × {recurrent, memoryless}, 30 updates, ppo-epochs 3, T=6 frames, 16 train / 16 held trajectories (disjoint seeds), N∈{8,12,16}, reconfig e_edge=0.1, cold-start. Activation confirmed per run (`mechanism_activation.json`: `dynamic_task.enabled=true`, `episode_length=6`, recurrence flag True/False per arm, `reconfiguration_cost_nonzero=true`).

### 6.1 Headline (held set)

| arm | held per-frame feasibility (per seed) | mean | 95% CI | mean episode return |
|---|---|---|---|---|
| **recurrent** (cross-frame hidden) | 0.156, 0.000, 0.000 | **0.052** | [−0.17, +0.28] | −4.81 |
| **memoryless** (hidden reset/frame) | 0.167, 0.000, 0.000 | **0.056** | [−0.18, +0.29] | −4.74 |
| **myopic-greedy reference** (reconfig-blind per-frame best of 6 variants) | 0.188, 0.177, 0.292 | **0.219** | [+0.06, +0.38] | −3.98 |
| **paired (recurrent − memoryless)** | −0.011, 0.000, 0.000 | **−0.004** | **[−0.018, +0.011]** | −0.072 (CI [−0.38, +0.24]) |

**The paired recurrent−memoryless difference spans 0 on both feasibility and return → cross-frame recurrence does NOT beat the memoryless history-aware actor.** This confirms the §3 pre-registered prediction. (Evidential-base caveat: see §6.3 — only 1 of 3 seeds trained successfully, so the spans-0 result rests largely on one informative seed plus two seeds where *both* arms collapsed identically.)

### 6.2 Per-seed training behaviour (the honest breakdown)

The three seeds split into two distinct outcomes (verified from the six `training_history.json`):

| seed | recurrent train→held feas | memoryless train→held feas | actor grad-norm (max) | outcome |
|---|---|---|---|---|
| 0 | 0.27 → 0.156 | 0.27 → 0.167 | 78 / 122 | **trained**; train→held generalization gap |
| 1 | 0.00 → 0.000 | 0.00 → 0.000 | 165 / 131 | **training collapse** (dead all-infeasible policy) |
| 2 | 0.00 → 0.000 | 0.00 → 0.000 | 439 / 536 | **training collapse** (dead all-infeasible policy) |

- **3-seed-mean train val feasibility = 0.090 (identical for both arms)**, NOT ~0.27 — that 0.27 is **seed-0 only**. Held mean = 0.052 / 0.056.
- On **seed 0** both arms learn (train val ~0.27, ≈ the myopic reference ~0.22), critic EV rises to ~0.8, and both **reduce switching** (switches/frame 1.2→0.36 — learning to hold topologies); held drops to ~0.16 → a **genuine generalization gap**.
- On **seeds 1 & 2** both arms **fail to train at all**: val feasibility = 0.0 at *every* update on train *and* held, return = −5.4 = −τ·6 (the dead all-infeasible policy), with actor grad-norm spikes to 165/439/536 (pre-clip; the applied update is clipped to norm 1.0). This is a **training/optimization instability, not a generalization gap** (nothing was learned to then fail to generalize).

### 6.3 The two failure modes (do not conflate them)

The low held feasibility (~0.05) is the average of **two different failures**: one seed with a real train→held **generalization gap** (0.27→0.16, the same N≤16 bottleneck the static campaign named in Phase 12), and two seeds with an **optimization collapse** (cold-start dynamic recurrent-PPO diverges to a dead policy). Recurrence helps neither: rec ≈ mem on the trained seed (0.156 vs 0.167) and rec == mem (both 0.0) on the collapsed seeds. The myopic-greedy reference (a non-learned per-frame reactive policy) generalizes better (held 0.22) than the learned RL on every seed — consistent with the campaign's "simple baseline wins at N≤16" pattern.

---

## 7. Reproduction (driver / command / config / manifests)

```bash
# Data manifest (provenance: seeds, splits, distributions, content hashes)
PYTHONPATH=src python scripts/diagnostics/dynamic_data_manifest.py --seeds 0 1 2 \
  --dyn-train 16 --dyn-held 16 --frames 6

# Temporal Value Test (Delta_H) on real mobility data
PYTHONPATH=src python scripts/diagnostics/dynamic_temporal_value.py --count 24 --frames 8

# One dynamic run (recurrent arm) -- the trunk --dynamic branch
PYTHONPATH=src python scripts/train/train_decentralized_rl.py --dynamic --cold-start \
  --reward-mode dense --dynamic-actor recurrent --reconfig-e 0.1 --updates 30 --ppo-epochs 3 \
  --dyn-train 16 --dyn-held 16 --frames 6 --dyn-eval-every 8 --seed 0 --out-dir <dir>

# Full headline (recurrent vs memoryless x 3 seeds + myopic reference)
PYTHONPATH=src python scripts/diagnostics/dynamic_headline.py --seeds 0 1 2 --updates 30 \
  --ppo-epochs 3 --dyn-train 16 --dyn-held 16 --frames 6 --dyn-eval-every 8 --reconfig-e 0.1
```

**Config (D4 headline):** operating-point urban v2x_37885 regime (tx 20 dBm, 4 RSU, relay-3, shadowing+NLOSv, coverage-gated); N∈{8,12,16}; T=6 frames; dt=2 s; speed 15–30 m/s; hold_interval=4; γ=0.95; reconfig e_edge=0.1; reward dense; cold-start; per-agent BCSP + local_mutual_assemble; per-frame centralized critic; keep-best on a periodic train-decoded eval (never the held set). Artifacts: `result_save/_dyn_headline/{arm}_seed{n}/` + `result_save/dynamic_headline.json` + `result_save/dynamic_data_manifest.json` + `result_save/dynamic_temporal_value.json`.

---

## 8. Honest analysis & verdict

**What was delivered (all owner requests met).** `two_timescale_env` is wired into the trunk (`--dynamic`); genuine multi-frame channel data is generated (moving-vehicle trajectories on the production channel, with predictable link evolution); the full episode-recurrent model is trained and tested with the complete instrumentation set (data manifest, mechanism activation, training logs, held state-transition traces, per-seed/CI, the deployed-decoder eval). The dynamic mechanism is genuinely **active** (verified by `mechanism_activation.json` per run), not an inert opt-in.

**What the test shows — three honest conclusions:**

1. **Cross-frame recurrence provides no gain over a memoryless history-aware actor.** Paired recurrent−memoryless = −0.004 (feasibility, CI [−0.018, +0.011]) and −0.072 (return, CI [−0.38, +0.24]) — both span 0; rec ≈ mem on the one trained seed (0.156 vs 0.167) and rec == mem (both 0.0) on the two collapsed seeds. This is consistent with the pre-registered §3 prediction (with full per-frame CSI the task is ~Markov in (current channel, previous topology), both observed by the memoryless actor) and with the Temporal Value Test's bounded headroom (median Δ_H=0; ~21% of scenes positive). **Evidential-base caveat:** only 1 of 3 seeds trained, so this rests largely on one informative seed plus two seeds where both arms collapsed identically — directionally firm and theory-consistent, but a multi-seed claim needs the stability fix below.

2. **Two distinct failure modes keep absolute performance low — and neither is relieved by temporal structure.** (i) *Generalization gap* (seed 0): both arms learn on train (~0.27 ≈ myopic) but drop to ~0.16 on held — the same N≤16 generalization bottleneck the static campaign named (Phase 12). (ii) *Training instability* (seeds 1 & 2): cold-start dynamic recurrent-PPO diverges to a dead all-infeasible policy (val 0.0 throughout, grad-norm spikes 165–536 pre-clip). The learned RL under-performs even a non-learned per-frame greedy (held 0.22) on every seed. Temporal modeling addresses neither.

3. **The campaign pattern extends to the temporal axis.** Across the v2 campaign, every sophisticated mechanism (COMA counterfactual credit, SCQ, chance/CVaR/Pareto, PNA actor) was verified-correct but did not beat the simple baseline at N≤16 single-step. The **temporal/recurrent actor is now the seventh such mechanism**: correct, genuinely active, but no headline gain over the simpler (memoryless) variant.

**Honest caveats / scope.** (a) The recurrent-vs-memoryless comparison is FAIR (identical architecture/budget/data; only cross-frame memory differs), but the absolute level is limited by the generalization gap (seed 0) and the cold-start instability (seeds 1 & 2), not by temporal structure. (b) e_edge=0.1 is the regime where the Temporal Value Test shows non-trivial Δ_H; at e_edge=0 switching is free and the task is trivially myopic. (c) Heavy O(n⁴) N=16 evaluation caps the budget (30 updates, 16 trajectories). **The path to a firmer claim** is: stabilize cold-start dynamic PPO (lr/entropy schedule, reward scaling, or a BC warm-start for trajectories) and scale seeds + train trajectories. The §3 theory + the seed-0 rec≈mem result predict this would raise the absolute level but **not** overturn the temporal-mechanism conclusion.

**Verdict:** the dynamic two-timescale task is now built, wired into the trunk, and honestly tested with full instrumentation. **The temporal/recurrent mechanism is verified-correct and genuinely active but yields no measured gain over a memoryless history-aware actor; the dynamic task under full per-frame CSI is effectively Markov, and the binding limits are held-set generalization and cold-start training stability, not temporal modeling.** Recommendation: keep the dynamic arm **opt-in** (`--dynamic`, default off; T=1 path byte-identical), consistent with every other v2 mechanism, and treat dynamic-PPO stabilization + larger-scale generalization as the open frontier.

---

## 9. Adversarial verification

A 4-lens adversarial Workflow (11 agents, 25 findings, 6 deep refutation passes — 0 refuted) independently re-verified the build. **Outcome: no blockers.** It read the real code/artifacts and re-derived every load-bearing claim:

**Confirmed correct (independently reproduced):** each frame is a real Stage-21 channel measurement on `advance_scene`-moved geometry (not faked); node set + edge-id action space invariant across frames; reconfig cost = (e+l)·|△| with no t=0 charge; frame 0 == static scene; manifest determinism/hashes/disjoint geometry genuine; held eval uses the deployed torch-free `local_mutual_assemble` (no global decoder anywhere — `global_argsort_assemble` absent from the rollout); actor uses only local features + neighbour messages; critic training-only; `--dynamic` returns before the first T=1 statement → T=1 byte-identical, `reward_of` passed in (single objective, no fork); scaled-tanh preserves the logit≥0 gate; return recursion + `A_t=G_t−V` correct; PPO ratio per-agent (not joint); BPTT genuinely flows cross-frame (probe: recurrent grad 1.11 vs memoryless 0.0); the ablation is fair (only `hidden = h_next if recurrent else None` differs); keep-best on train-decoded val, never held; the Temporal-Value DP is exact and the `b152bdd` candidate-set fix is real; headline JSON ↔ report numbers match; no smoke params smuggled into the headline.

**MAJOR (fixed in this revision):** the original draft framed the low held feasibility as a single "generalization gap" using a **seed-0-only** train numerator (~0.27) against a 3-seed-mean held denominator (~0.05). The verifier showed seeds 1 & 2 are 0.0 on *train and* held (a training collapse, not a generalization gap) and the true 3-seed-mean train val is 0.090. §6.2/§6.3/§8 now report the two failure modes separately and like-for-like.

**Nits recorded (honest):** (N1) §3 stale "ceiling ~0.5" → corrected to the measured ~0.22. (N2) the spans-0 / "Markov" conclusion rests on 1 informative seed → caveat added (§6.1, §8.1). (N3) "rollout uses the deployed decoder" was imprecise — the rollout uses the matched stochastic BCSP sampler (MAP == the decoder); the deployed `local_mutual_assemble` is used at *eval*. Docstring + `mechanism_activation.json` action field corrected. (N4) `hold_interval` scales the per-frame objective only in the sibling `two_timescale_env` (the temporal diagnostic), not in the RL reward (`reward = base − reconfig`, single-round base); ablation-neutral, noted for the owner. (N5) the manifest's `scenario_ids` are name-by-index (identical strings across splits though geometry is disjoint) and `solvability_family` is the intended label suffix (not measured); `observation()['label']` hard-codes `feasible_exists=True` (dead metadata — reward reads feasibility live, so data is uncorrupted). (N6) `test_recurrent_reroll_bptt_reaches_gru` proves GRU reachability but passes in both modes — it does not discriminate cross-frame carry (the capability is real per the N3 probe); a late-frame-only discriminating test is the suggested follow-up.

The verifier's bottom line: data and decentralization are sound, the "recurrence shows no gain" conclusion is not compromised, and with the MAJOR re-aggregation + N1 fix applied (done here) the build is a PASS.
