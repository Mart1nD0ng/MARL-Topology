# R4 — experiment plan (beneficial oracle-edit dataset) — the PIVOTAL stage

## Hypothesis (the single thing R4 answers)
R3 showed the trainer is stable but stays at the anchor because there is no beneficial-edit DIRECTION signal.
R4 measures whether such a signal EXISTS: over the deployable `local_hysteresis` anchor `x_H`, do single LOCAL
edits (add / remove / swap) with a POSITIVE true-objective gain (`ΔJ > margin`, safety-ok) exist at a
non-trivial rate? If yes → R5 can supervise them. **If the positive-edit rate is ~0, the anchor is near a
local optimum in the local-edit space → there is nothing to learn → the belief-guided residual-PPO route is
an honest campaign-closing NEGATIVE at N≤16 (all four chains non-load-bearing).**

## Single change (a teacher dataset; NOT a deployment mechanism)
`src/marl_topology/training/oracle_edit_dataset.py` + a generator. For each frame, with the anchor `x_H`:
- enumerate LOCAL edits: **add** (each non-anchor edge, budget-feasible), **remove** (each anchor edge),
  **swap** (bounded: top-k add × top-k remove pairs).
- with the TRAINING evaluator (true current channel + closed-form whole-network quorum tail) compute per edit:
  `ΔC = C(x_H⊕e)−C(x_H)`, `ΔD_quorum`, `ΔE`, `ΔL`, and `ΔJ = reward_of(x_H⊕e) − reward_of(x_H)` (the SAME
  dense reward the RL uses — `J` consistent with the reward).
- `safe = NOT (C(x_H)≥τ AND C(x_H⊕e)<τ)` (the edit must not turn a feasible anchor infeasible).
- `positive = (ΔJ > margin) AND safe` (margin reported, e.g. 0.0 and a small 0.01 variant).
- store ONLY `(anchor, the single edit descriptor, the Δ's)` — **never the full oracle topology** (TechSpec §11).

## Metrics (urban + random, ≥5 seeds, CI)
`positive_edit_rate` (per (frame,edit)); `best_edit_gain` distribution (max ΔJ per frame);
`anchor_failure_repairable_rate` (anchor-infeasible frames where some add/swap reaches feasibility);
`safe_prune_rate` (anchor-feasible frames where some remove cuts energy without breaking feasibility);
`evaluator_calls`.

## Mechanism-Path Matrix delta
- beneficial-edit dataset: NOT_PRESENT → IMPLEMENTED (teacher-only; never at deployment). Becomes the
  supervised target for the R5 repair/safety/edit heads (ACTIVE_IN_LOSS at R5).

## Failing-first tests (`tests/unit/test_belief_residual_R4_oracle_edit.py`) — fail on HEAD
- `test_oracle_edit_dataset_contains_anchor_and_delta` — each record has the anchor + a single edit + the Δ's.
- `test_only_positive_gain_edits_supervised` — the supervised subset is exactly `{ΔJ>margin ∧ safe}`.
- `test_edit_labels_include_C_D_E_L` — each record carries ΔC, ΔD_quorum, ΔE, ΔL, ΔJ.
- `test_dataset_artifact_hash_logged` — the dataset writes a content hash (provenance).
- `test_teacher_does_not_use_held_for_training` — the generator takes ONLY train scenes (held never passed).
- `test_no_full_oracle_topology_stored` — records store the edit descriptor, not a full edited topology.

## Exit condition / decision rule
- positive_edit_rate (5-seed CI) **> 0 non-trivially** AND `anchor_failure_repairable_rate` or
  `safe_prune_rate` > 0 → KEEP → R5 (train heads on the positive edits).
- positive_edit_rate ≈ 0 (CI ~0) → **STOP RL**: path-specific negative — the anchor is near local-optimal;
  the local-edit space has no learnable beneficial direction at N≤16. (Honest campaign close.)

## Out of scope
Training the heads (R5); deployment gating (R6). R4 only builds + measures the teacher signal.
