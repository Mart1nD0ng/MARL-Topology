# CURRENT_DYNAMIC_REPAIR_STATUS — HEAD vs. Contract-v3 + Dynamic-Repair-Plan

> Document type: gap analysis / repair-campaign status of record.
> Authority: `docs/MARL-Topology-Development-Contract-v3.md` (the contract) and
> `docs/MARL-Topology-Dynamic-Repair-Engineering-Plan.md` (the plan) are the highest-priority
> binding documents. Where this file, old code/comments, or any report conflicts with those two,
> **they win**.
> Iron rule (inherited from `CURRENT_HEAD_STATUS.md`): every claim cites an artifact
> (file:line, test id, command output). Anything not so grounded is marked `UNVERIFIED`.

| field | value | source |
|---|---|---|
| git HEAD | `e450cb7` "D6: stabilized warm-started 3-seed headline + report update" | `git rev-parse HEAD` |
| branch | `decentralized-marl-trunk` | session git status |
| date | 2026-06-26 | session |
| dynamic entry | `scripts/train/train_decentralized_rl.py --dynamic` → `training/dynamic_rl.run_dynamic_training` | code |

This file answers the loop's first task: **does HEAD differ from the contract + the plan, and where?**
It is grounded in direct reads of the dynamic code path (not the reports). It is a frozen snapshot
to drive the D0→D14 repair; it is not an architecture redesign.

---

## 0. Method

The dynamic task is reached only via `--dynamic` (`train_decentralized_rl.py`) → `run_dynamic_training`
(`src/marl_topology/training/dynamic_rl.py`). Its data comes from `sample_dynamic_scenes`
(`src/marl_topology/training/dynamic_frames.py:224`). Every claim below was read from those files +
`two_timescale_env.py`, `stage31_scenario_generator.py`, `graph_payload.py`, and the production
evaluator. The 10 loop-flagged gaps are each confirmed or refuted against code.

---

## 1. The 10 flagged gaps — code-grounded verdicts

State vocabulary per Contract §1.

| # | Claim under test | Verdict | Code evidence | Contract / Plan ref | Fix stage |
|---|---|---|---|---|---|
| 1 | Dynamic data is single-RSU random geometry, **not** 4-RSU urban grid | **RESOLVED (D1)** | `sample_dynamic_urban_scenes` now builds real 4-RSU urban scenes via `build_urban_grid_scene` (4 RSU + G×G buildings + street grid + road-constrained motion); `--dyn-data urban` opt-in; provenance manifest + content hash verifies the description against the source. The single-RSU `sample_dynamic_scenes` is kept as the `dynamic_random_geometry` ablation. | Contract §4.1, §16.8; Plan D1 | **D1 ✓** |
| 2 | `hold_interval` does **not** enter the RL reward; Temporal-Value-Test uses a different objective | **CONFIRMED gap** | RL: `dynamic_rl.episode_rollout:94-97` → `reward = base_r − reconfig` (no `H` factor). TVT: `two_timescale_env.step:104` → `base_objective = cost_fn(...) * hold_interval`. The two objectives differ by the `H` multiplier. (Report §9 N4 already admits this.) | Contract §3.2, §3.3, §16.9; Plan D2 | **D2** |
| 3 | Training optimizes discounted return but eval/keep-best uses undiscounted return | **CONFIRMED gap** | Train advantage + critic target are discounted `G_t` (`dynamic_rl.py:105-108`). But `dynamic_eval` accumulates **undiscounted** `ep_ret += base_r − reconfig` (`:161`), and keep-best uses that (`val_score = val["mean_episode_return"]` `:390`, `best_val` `:406-407`). Train ≠ eval objective. | Contract §3.3, forbidden §13.9; Plan D2 | **D2** |
| 4 | No independent validation trajectories; checkpoint selected on train | **CONFIRMED gap** | Only two splits built: `train_scenes` seed`*1000+1` (`:256`) and `held_scenes` seed`*1000+777` (`:260`). **No `val_scenes` (seed`*1000+333`).** Keep-best runs `dynamic_eval(actor, train_scenes, …)` (`:387`) → checkpoint chosen on **train** eval. Report §7 confirms "keep-best on a periodic train-decoded eval". Per Contract §3.4 this makes the result *pilot-only, not headline-eligible*, and per forbidden §13.7 the seed-0 "generalization gap" framing (train keep-best → blame held drop on generalization) is itself disallowed. | Contract §3.4, forbidden §13.7/§13.10; Plan D3 | **D3** |
| 5 | Phase-specific PBFT message accounting is a primitive, **not** wired into the production evaluator | **RESOLVED (D4)** | `build_pbft_message_plan` now drives `_phase_specific_phase_records` in BOTH `evaluate()` methods (canonical + vectorized); opt-in `phase_specific_accounting`, activated in `operating_point_regime`. Energy/latency only; reliability untouched. | Contract §2.1, §5.3, §16.6; Plan D4 | **D4 ✓** |
| 6 | Dynamic branch has no COMA/SCQ/chance/CVaR/Pareto/PNA wired | **CONFIRMED gap** | `dynamic_rl.py` imports only `CentralizedGraphCritic`, `DynamicRecurrentActor`, BCSP action, `local_mutual_assemble`, and `graph_mappo` (V-critic value + PPO-clip). No `--counterfactual`, `--scq`, `--chance`, `--pareto`, `--actor pna` reach this path. Report §8.4 correctly states "tested dynamic recurrent-PPO only, not full Phase 8-11". | Contract §16.7; Plan D9–D12 | **D9–D12** |
| 7 | Actor observation lacks velocity/heading/relative-velocity/CSI-derivative → "Markov" claim unproven | **RESOLVED (D5)** | `--motion-features` (opt-in) appends LOCAL node velocity/heading + edge relative-velocity/distance-delta/csi-delta to `dynamic_frames.observation`; `graph_payload` untouched (static byte-identical). The Markovness A/B (current-CSI vs velocity) is now testable in D8. | Contract §3.5; Plan D5 | **D5 ✓** |
| 8 | myopic-greedy is a **central reference** (calls evaluator over named candidates), not a deployable baseline | **RESOLVED (D7)** | `training/dynamic_baselines.py` adds fair DEPLOYABLE baselines (local_threshold/hysteresis — local-only action, 0 evaluator calls) and groups the myopic-greedy as a CENTRAL reference; `baseline_budget_report` records per-method action-evaluator-calls. Diagnostic smoke: deployable BEATS the central myopic ref → the prior "simple baseline wins" framing must not conflate them. | Contract §10.1, §16.10, forbidden §13.11; Plan D7 | **D7 ✓** |
| 9 | Warm-start is plain per-edge BCE, not decoder-aware; no teacher-KL/BC anchor protects it during PPO | **RESOLVED (D6)** | `--dyn-warmstart-mode bcsp` = decoder-aware BCSP-subset NLL of the teacher proposals (budget-capped, mutual-decode); `--dyn-bc-anchor` = annealed teacher-BC anchor in the PPO loss; `--dyn-critic-warmstart` = critic value warm-start. Reports teacher source/leakage/warmstart-alone-return/post-RL-drift. | Contract §10.3; Plan D6 | **D6 ✓** |
| 10 | 3 seeds / 24 trajectories / 30 updates is diagnostic, not a final-failure headline | **CONFIRMED** | Report §6 config: 3 seeds, 24 train/24 held, 30 updates, 1/3 cold-start seeds trained. Per Contract §0, §13.15-16: must be `ACTIVE_IN_PILOT`/diagnostic, not `VALIDATED_NEGATIVE` for "dynamic MARL". | Contract §0, §12, §13.15-16; Plan D8/D13 | **D8, D13** |

**Net:** all 10 flagged gaps are real and confirmed in code. The DYNAMIC_TASK_REPORT.md is *largely
honest* about #6, #8, #10 (it scopes them), but its **data description (§6/§7 "4 RSU urban") contradicts
the generator code (#1)**, and #2/#3/#4 are口径 (objective-consistency) defects that make the current
recurrence/RL conclusions diagnostic, not final.

---

## 2. What is already correct (do not re-do)

Grounded, to avoid re-building working pieces:

- **T>1 episode rollout, BCSP per-agent action, `local_mutual_assemble` eval decoder, per-agent PPO
  ratio, per-frame centralized critic, discounted returns in training, cross-frame recurrent vs
  memoryless ablation** are all genuinely wired (`dynamic_rl.py`) and the `--dynamic` branch leaves
  the T=1 path byte-identical (verified by the prior adversarial pass, report §9).
- **Reconfiguration cost** `(e_edge+l_edge)·|E_t △ E_{t−1}|` with no t=0 charge is correct
  (`dynamic_rl.py:95-97`, `two_timescale_env.cost:46`).
- **Train==deploy decoder / decentralization (D1)**: rollout uses the matched stochastic BCSP whose
  MAP limit is the deployed torch-free decoder; critic is training-only.
- **Mechanism activation + instrumentation** (`mechanism_activation.json`, `training_history.json`,
  `held_traces.json`, `dynamic_result.json`) are emitted per run (`dynamic_rl.py:417-447`).
- **Data manifest + content hashes** exist (`result_save/dynamic_data_manifest.json`).

The repair is therefore mostly about **objective口径 (D2/D3), data realism (D1), observation
sufficiency (D5), warm-start protection (D6), fair baselines (D7), and mechanism coverage
(D4/D9–D12)** — not a from-scratch rebuild.

---

## 3. Hard-constraint (loop) compliance snapshot

| constraint | status at HEAD | note |
|---|---|---|
| Deployment fully decentralized (no critic/global decoder/evaluator at inference) | **HOLDS** | critic + teacher evaluator are training-only; eval uses `local_mutual_assemble`. |
| Training-only centralized critic/evaluator clearly marked | **HOLDS** | `CentralizedGraphCritic` training-only; teacher uses evaluator (training-only). |
| Rollout == deploy local mutual-acceptance semantics | **HOLDS** | MAP(BCSP) == decoder up to tie set. |
| `τ ≥ 0.9`, same reliability def train+eval | **HOLDS** | τ=0.9 constant; same `reward_of`. |
| Final reliability = closed-form whole-network PBFT quorum-tail | **HOLDS** | `protocol/quorum_tail.py`. |
| **Same dynamic objective across train / TVT / val / held** | **PARTIAL** | gaps #2 (H factor) + #3 (discount) **RESOLVED in D2** (`6→` commit); #4 (no val split) still open → D3. |
| `hold_interval·base − reconfig` reward, or stated equivalent | **RESOLVED (D2)** | `episode_rollout`/`dynamic_eval`/`_myopic_reference` now `H·base − reconfig`, discounted; TVT already matched. |
| train/val/held three-split; held excluded from checkpoint | **RESOLVED (D3)** | `--dyn-val` seed`*1000+333`; keep-best on val only; held final-only; `split_manifest.json` records it; 4-lens adversarial PASS (no leakage). |
| Data description matches source | **VIOLATED** | gap #1 (single-RSU vs "4-RSU urban"). → D1. |
| Central evaluator-greedy labeled reference, not deployable baseline | **PARTIAL** | gap #8. → D7. |
| default-off mechanisms not called "full model tested" | **HOLDS** | report §8.4 scopes this. |
| Every mechanism has runtime activation artifact | **HOLDS** | `mechanism_activation.json`. |

---

## 4. Stage status (D0–D14)

| stage | scope | status |
|---|---|---|
| **D0** | Freeze current dynamic results as the diagnostic baseline | **DONE** (commit `6fafd2d`/`7f238d4`) |
| **D1** | 4-RSU urban-grid dynamic data (roads/buildings/RSUs/motion/stable edge-ids/real Stage-21 frames) | **DONE** — `sample_dynamic_urban_scenes` (real `build_urban_grid_scene`: 4 RSU at intersections + G×G buildings + street grid; `_road_constrained_motions` axis-aligned grid-street constant velocity; stable ids; frame-0 static; channel evolves). `--dyn-data {random,urban}` opt-in (default random byte-identical); `dynamic_data_manifest.json` provenance + content hash. 6 tests; unit 703/0; adversarial verify PASS (genuinely 4-RSU urban, honest manifest, no over-claim). Urban A/B = D13. See `docs/dynamic_repair/D1/`. |
| **D2** | Dynamic reward: `hold_interval` into base; one discounted objective across train/TVT/val/held | **DONE** — gaps #2,#3 fixed (`episode_rollout`/`dynamic_eval`/`_myopic_reference` = `H·base−reconfig`, discounted; activation logs it; 4 failing→passing tests; suite 654/0, contract 63/0; smoke exit 0). Headline deferred to post-D3. See `docs/dynamic_repair/D2/`. |
| **D3** | train/val/held three-split; checkpoint by val discounted return only | **DONE** — gap #4 fixed (`--dyn-val` seed`*1000+333`; `build_split_manifest`; keep-best on val only; held final-only; pilot fallback labeled). 4 failing→passing tests + pilot-fallback test; unit 657/0, contract 63/0, smoke exit 0; **4-lens adversarial verification all PASS** (no held leakage, splits computed-disjoint, T=1 byte-identical). See `docs/dynamic_repair/D3/`. |
| **D4** | Wire phase-specific PBFT message plan into the production Stage-21 evaluator | **DONE** — gap #5 fixed (`phase_specific_accounting`: pre_prepare=primary star, prepare/commit=validator vote, clients excluded; energy/latency only, reliability untouched; activated in `operating_point_regime`; **also fixed in the VectorizedStage21Evaluator** the dynamic path uses — adversarial verify caught it inert otherwise). Corrected protocol energy 31–58% lower; reliability byte-identical. 8 new tests; unit 666/0, contract 63/0. See `docs/dynamic_repair/D4/`. |
| **D5** | Add velocity/heading/relative-velocity/CSI-delta/CSI-age to actor obs; training-only critic state | **DONE** — gap #7 fixed (`--motion-features` opt-in: node velocity/heading + edge relative-velocity/distance-delta/csi-delta; LOCAL/neighbour-only; `graph_payload` untouched → static byte-identical; critic gets global velocities training-only). 5 tests; unit 671/0, contract 63/0; adversarial verify PASS (locality confirmed). See `docs/dynamic_repair/D5/`. |
| **D6** | Decoder-aware BCSP-subset warm-start + teacher-KL/BC anchor; critic warm-start | **DONE** — gap #9 fixed (`--dyn-warmstart-mode bcsp` decoder-aware BCSP-subset NLL; `--dyn-bc-anchor` annealed teacher anchor in PPO; `--dyn-critic-warmstart`; reports teacher source/leakage/warmstart-alone-return/post-RL-drift). 4 tests; unit 675/0, contract 63/0; adversarial verify PASS (no leakage). A/B pilot deferred to D8. See `docs/dynamic_repair/D6/`. |
| **D7** | Fair deployable baselines + separate central-reference group | **DONE** — gap #8 fixed (`training/dynamic_baselines.py`: local_threshold/local_hysteresis deployable policies = local-only action, 0 evaluator calls, mutual-acceptance budget-respecting; `evaluate_central_reference` = myopic-greedy, action calls evaluator, grouped separately; `baseline_budget_report`). 3 tests; unit 678/0, contract 63/0; adversarial verify PASS. Smoke (diagnostic): deployable beats the central myopic ref. See `docs/dynamic_repair/D7/`. |
| **D8** | Re-test recurrent vs memoryless across {current-CSI, velocity, recurrent, velocity+recurrent} | **DONE** — 5-seed N∈{8,12,16} 2×2 headline on the corrected pipeline. **All paired diffs span 0** → velocity AND recurrence give no significant gain. Learned arms 0.05–0.20 < deployable baselines 0.36–0.37 < central myopic 0.37 → binding limit is RL learning, not temporal. D6 validated: post-RL drift small/+ve, no seed collapse. Adversarial re-derivation PASS. SCOPE: single-RSU (NOT urban; D1 pending). See `docs/dynamic_repair/D8/`. |
| **D9** | Dynamic COMA / Q-critic (per-agent counterfactual credit) | **DONE** — `--counterfactual` opt-in: action-conditioned Q critic Q(s_t,S_t)→G_t; per-agent A_{i,t}=Q−E[Q(S̃_i,S_{-i})] via reused static COMA machinery; budget-neutral (critic forwards, 0 extra evaluator calls); default-off byte-identical. 4 tests; unit 684/0; adversarial verify PASS. A/B deferred to D13. See `docs/dynamic_repair/D9/`. |
| **D10** | Dynamic SCQ (closed-form one-step counterfactual supervision of the Q critic) | **DONE** — `--scq` opt-in (requires `--counterfactual`): target Δy=ΔR+γ(V(s_{t+1})−V(s̃_{t+1})) (one-step return; bootstrap free; distinct from static ΔR); SCQ loss enters the critic only; NOT budget-neutral — `scq_evaluator_calls_per_update` reported. 4 tests; unit 688/0; adversarial verify PASS. A/B deferred to D13. See `docs/dynamic_repair/D10/`. |
| **D11** | Chance/CVaR/Pareto into the dynamic task | **DONE** — `dynamic_reliability.py`: `--chance` sign-flexible episode dual (budget-neutral, penalty in train+eval); CVaR tail metric (budget-neutral); `--pareto-archive` checkpoint selection from VAL by reliability→non-dominated (NOT raw feasibility; extra eval reported, NOT budget-neutral); preference primitive verified (for D12). 4 tests; unit 692/0; adversarial verify PASS. A/B deferred to D13. See `docs/dynamic_repair/D11/`. |
| **D12** | PNA/recurrent PNA dynamic actor (+ preference-conditioned) | **DONE** — `models/dynamic_pna_actor.py` (PNA directional + cross-frame GRU + ω-preference; drop-in for DynamicRecurrentActor); `--dynamic-actor-arch pna` opt-in (orthogonal to recurrence); DEPLOYED-path → DECENTRALIZED (local+neighbour+public ω, no global state), NaN-safe, default-mlp byte-identical. 5 tests; unit 697/0; adversarial verify PASS. A/B deferred to D13. See `docs/dynamic_repair/D12/`. **All Phase-8–11 mechanisms now in dynamic.** |
| **D13** | Multi-seed / multi-N / multi-param dynamic campaign with CI + scope | **DONE** — `dynamic_d13_campaign.py` 5-seed × 8-arm × N{8,12,16} on BOTH real 4-RSU urban (`--dyn-data urban`) and single-RSU random. HONEST NEGATIVE: every mechanism A/B (recurrent+velocity / COMA / SCQ / chance / Pareto / PNA) paired-vs-baseline feasibility CI spans 0 (no significant gain at 5 seeds); learned arms (0.15–0.32) < deployable heuristics (0.37 random / 0.71 urban) < central myopic (0.37 / 0.79) on both data → binding limit = RL feasibility-region learning. PNA largest +0.125 but bimodal (seed-0 collapse), not significant; chance hurts return (−35.9, λ→4.46). Budget honest (SCQ 43/upd, Pareto 144 non-neutral reported). Adversarial re-derivation (4 lenses from raw JSON, incl. byte-identical urban-hash regeneration) PASS. See `docs/dynamic_repair/D13/`. |
| **D14** | Docs / report / README / AGENTS reconciliation | **DONE** — SUPERSEDED banner on `DYNAMIC_TASK_REPORT.md` (frozen=single-RSU not 4-RSU urban; "full model"=recurrent-vs-memoryless only; D2–D7 corrections); dynamic-repair-complete banner on `CURRENT_HEAD_STATUS.md`; `README.md` rewritten critic-free-identity→CTDE (recommended `--baseline graph-mappo`, dynamic arm documented, stale headlines retired, I1→deployment-decentralization); consolidated D0–D14 SUMMARY + stage/mechanism ledger in `URBAN_V2X_RESEARCH_LOG.md`. Claims-vs-evidence verify (5 items A–E) PASS, exact numeric match to D13. Suite 706 unit/63 contract 0-fail. See `docs/dynamic_repair/D14/`. **CAMPAIGN COMPLETE.** |

**Execution order (Plan §17, compute-limited):** D0 → D2 → D3 → D5 → D6 → D7 → D8, with D1 and D4
proceeding in parallel but mandatory before any final headline.

---

## 5. Conclusion scope

This status proves only **where HEAD diverges from the contract + plan**. It does **not** re-run any
experiment or revise any result. The existing dynamic negative result (recurrence ≈ memoryless; RL ≤
myopic teacher) is **diagnostic and scope-limited** — it was measured under (a) single-RSU random
geometry (not urban), (b) a reward without `hold_interval`, (c) a train/eval objective mismatch, (d)
no validation split, (e) an observation without motion features, and (f) without any Phase-8–11
mechanism — so it cannot be promoted to "dynamic MARL failed". The repair plan D0–D14 fixes each
defect before any such claim is admissible.
