# Q1 — decision (stale / partial CSI observation model)

**Result: KEEP.** A new, opt-in, deterministic stale/partial CSI observation model lets the dynamic
actor decide on a lagged/held channel `ĝ_t` while the reward / evaluator / reliability stay on the TRUE
current channel `g_t`. Default (`--csi-mode current` / no model) is byte-identical. Adversarially
verified (independent agent, 7 claims) = **PASS** (one MINOR doc nit, fixed). This is the Axis-A
precondition; Q2 measures whether memory recovers value (no temporal A/B claimed here).

## What changed (opt-in, default-off byte-identical)
- New `src/marl_topology/training/csi_observation_model.py`: pure `CsiObservationModel`
  (current|delay|partial|delay_partial; deterministic `hashlib` probe mask; unified plan
  `src = max(0, last_probe − delay)`, `age = t − src`, frame-0 floor; optional logit-noise default 0).
- `DynamicScene.csi_observation_model` field + `_apply_stale_csi`: overwrites channel CSI cols {0,1,2,3}
  (psucc/psucc/latency/energy) with the OBSERVED value read from the cached true `context(src)`, and
  appends `[csi_age, csi_observed_mask]`. Gated by `is_active()` → current/None untouched.
- Threaded through `dynamic_scene_from_motion` / `sample_dynamic_scenes` / `sample_dynamic_urban_scenes`.
- Trunk: `build_csi_observation_model(args)` + `--csi-mode/--csi-delay/--csi-probe-prob/--csi-noise-std/
  --csi-seed`; `csi_observation` activation block (mode, age stats, observed fraction, invariant flags).

## Verification (math → integration → smoke → activation)
- **Unit (9 new, all pass):** probe determinism; source/age formula; partial hold-last + age increment;
  current byte-identical (`torch.equal`); delay-1 = previous frame; **no-leak** (the evaluator metrics are
  identical with/without staleifying); actor ≠ true current under stale.
- **Full unit suite: 715 passed / 0 failed** (706 → 715).
- **Real-shard smokes (exit 0):** `--csi-mode delay --csi-delay 1` → activation `mean_csi_age 0.75`,
  `max 1`, `stale_edge_frame_fraction 0.75`, `observed_fraction 1.0`; `--csi-mode partial
  --csi-probe-prob 0.5` → `observed_fraction 0.64`, `mean_age 0.51`, `max_age 3` (hold-last accumulates).
- **Run-level byte-identity:** `--csi-mode current` gives held_feas 0.625 / return −0.24183, IDENTICAL to
  the default (no-flag) run; `current` block = `{mode: current, is_active: False}`.

## Adversarial verification (independent agent, 7 claims) — PASS
1. NO-LEAK ✓ — every reward/feasibility/chance/CVaR/quorum-tail/SCQ-Δ/Pareto value comes from
   `context.evaluator` (true channel) + decoded topo, never the staleified `ef`
   (`train_decentralized_rl.py:128-132,186-211`; `dynamic_rl.py` reward/SCQ/chance/Pareto paths).
2. CRITIC — reads the **stale** observation (input); its regression **target** is the true-channel
   reward. Spec S4.6 permits (not requires) a true-CSI critic → deferred CTDE option, not a violation.
3. BYTE-IDENTITY ✓ — `is_active()` gates everything; None/current append no columns, change no values.
4. NO FABRICATION ✓ — reads real `context(src).link_records` over a frame-invariant edge set; `link is
   None` branch unreachable; no invented values.
5. DETERMINISM ✓ — `hashlib.sha256` (cross-process stable), frame 0 always probed.
6. ACTIVATION HONESTY ✓ — gated; age/observed-fraction from the real plan; flags consistent with (1).
7. SHORTCUTS ✓ — none of Spec §16 / workflow-forbidden patterns present.
- MINOR (fixed): module docstring + activation comment now state precisely that the critic INPUT is
  stale while its TARGET is true-channel (added `critic_input_csi` / `critic_target_csi` activation keys).

## Honest scope notes (NOT headlines)
- The smoke shows stale CSI degrades a **memoryless** actor (0.625 → delay 0.44 / partial 0.125), as
  expected for a fresh POMDP. This is **2-update, 1-seed, pilot params** — NOT a result. Whether a
  recurrent actor recovers the loss is exactly the Q2 question (CSI-prediction + Temporal-Value A/B).
- Noise (`--csi-noise-std`) is implemented but default 0 and unused in any Q1 pilot (one variable).
- Per-node probe budget deferred (per-edge ρ is the Q1 primary, per Spec S4.3).

## Acceptance table (Contract v3 §15)
- Phase: **Q1 — stale/partial CSI observation model**
- Status: **VALIDATED_POSITIVE (mechanism correct + leak-free + activated)** — observation-model stage,
  not a performance claim.
- Implemented ✓ / Wired into entrypoint ✓ (`--csi-mode`) / Active by default ✗ (opt-in) / Active in this
  run ✓ (delay + partial smokes)
- Test scale: pilot smokes (N=8, 4 frames, 2 updates) + 9 unit + full suite 715/0; Seeds: 1 (smoke only)
- Dataset: `--dyn-data random` (single-RSU); Evaluator: closed-form whole-network quorum-tail on TRUE
  current channel; Decoder: local mutual-acceptance (unchanged); Critic: per-frame, reads stale obs
- Mechanisms active: stale/partial CSI observation. Mechanisms not tested: temporal value (Q2),
  D_quorum (Q3+), residual (Q5+).
- Positive findings: leak-free stale/partial CSI; deterministic; byte-identical default.
- Negative findings: none in scope (Q1 is an observation model, not a policy result).
- Conclusion scope: the observation model is correct and leak-free; NO temporal/performance claim.
- Next action: **Q2 — CSI-prediction + Temporal-Value health check** (memoryless vs recurrent, current
  vs stale; does history/velocity improve the estimate of the true current CSI under stale observation).
