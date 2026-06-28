# Q1 — Stale / Partial CSI observation model

## Hypothesis (the single thing this stage establishes)
The actor can be made to decide on **stale/partial CSI** `ĝ_t` while the reward / evaluator / critic
keep using the **true current CSI** `g_t`. This is the observation-side change that makes the dynamic
task a genuine POMDP (Axis A); Q2 then MEASURES whether history recovers value. Q1 itself proves only:
stale CSI changes the actor observation **and nothing in the reward/eval path** (no leak).

## Single change (one variable)
A new `src/marl_topology/training/csi_observation_model.py` + its wiring into the dynamic actor
observation. Modes: `current | delay | partial | delay_partial`, with CSI age + observed mask. Default
(no model / `mode=current`) is **byte-identical** to HEAD.

## Design (code-grounded)
- Edge feature layout (`data/graph_payload.py` L61-71): cols **0,1 = psucc**, **2 = latency_s**,
  **3 = energy_j** are the channel CSI; cols 4 (prev-topo), 5 (distance), 6-7 (roles) are NOT channel
  CSI and stay current. Only cols {0,1,2,3} are staleified.
- The observed source frame is a **pure, deterministic** function of `(edge_id, t)`:
  `src = max(0, last_probe(edge_id, t) − delay)`, `age = t − src`, `observed_now = probe(edge_id, t)`.
  - `current`: delay=0, prob=1 → src=t, age=0 (inactive → byte-identical).
  - `delay`: prob=1 → src=max(0, t−δ), age=min(t, δ).
  - `partial`: δ=0, probe ~ deterministic Bernoulli(prob) via stable hash (frame 0 always probed);
    unprobed → hold last observed value, age increments.
  - `delay_partial`: both.
- `DynamicScene.observation` looks up `self.context(src).link_records[edge_id]` (cached true channel of
  an earlier frame) to overwrite ef cols {0,1,2,3}, then appends `[csi_age, csi_observed_mask]`.
- **No-leak invariant (structural):** `reward_of` → `_evaluate` reads `sample["context"].evaluator`
  (`scripts/train/train_decentralized_rl.py` L130), never `ef`. Staleifying `ef` cannot change reward.

## Controlled variables
- Data: `--dyn-data random` (single-RSU) for the cheap pilot; `motion_features` OFF (clean single var).
- N small for pilot; the conclusion of Q1 is leak-freeness + activation, NOT a performance headline.

## Failing-first tests (must fail on HEAD, pass after)
- `test_current_mode_byte_identical` — model with mode=current (and model=None) → ef byte-identical.
- `test_delay1_observation_uses_previous_frame_csi` — delay=1: observed psucc(t)=true psucc(t−1); t=0 clamps.
- `test_partial_csi_holds_last_observation` — unprobed frame holds the last probed value.
- `test_csi_age_increments_when_unobserved` — age col +1 per unprobed frame, resets at a probe.
- `test_reward_uses_true_current_csi_not_observed_csi` — reward_of is identical with/without a stale model.
- `test_actor_never_receives_true_current_csi_under_stale_mode` — under delay≥1 + changing channel, the
  actor ef CSI ≠ true current context CSI on ≥1 edge/frame.
- math-unit: `test_csi_model_probe_mask_deterministic`, `test_csi_model_source_age_formula`.

## Success criterion (Q1 passes iff)
1. All failing-first tests pass; the affected unit suite stays green.
2. `--csi-mode delay --csi-delay 1` real-shard smoke exits 0; `mechanism_activation.json` records
   csi_mode/delay/probe + mean_csi_age>0 + observed≠true fraction>0 (nonzero signal).
3. The reward/held metrics are computed on true current CSI (no-leak test passes); current mode is
   byte-identical (a current-vs-HEAD run gives identical held metrics).

## Failure criterion
Any leak (reward/eval changes when only ef is staleified); current mode not byte-identical; the model
fabricates CSI not present in any past true frame; activation artifact missing or shows zero signal.

## Out of scope for Q1 (deferred)
Temporal-value measurement (Q2); noise (`noise_std`) implemented but default 0 and not in the pilot;
per-node probe budget (after per-edge ρ); D_quorum (Q3+).
