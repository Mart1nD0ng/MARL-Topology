# D5 — experiment_plan

**Stage:** D5 — add local motion features to the dynamic actor observation (Plan §7; gap #7).

**Hypothesis:** The dynamic observation is current-channel + previous-topology + step only — it has NO
velocity/heading/relative-motion, so the report's "~Markov in (current CSI, previous topology)" claim is
unproven (Contract §3.5). Adding LOCAL motion features (each node's own velocity/heading; per-link
relative velocity / distance-delta / CSI-delta) gives a memoryless actor the information to predict the
next frame's link evolution — making the memoryless-with-velocity arm a fair test of Markovness (D8).

**Single change (one variable = the motion features in the observation), opt-in:**
- `DynamicScene` carries per-node `velocities` (from the `NodeMotion`s), `dt_s`, and a `motion_features`
  flag. `observation(t)` appends, ONLY when `motion_features`:
  - node: `[velocity_x, velocity_y, speed, heading_sin, heading_cos]` (each node's OWN velocity — local).
  - edge `(u,v)`: `[relative_velocity_along_link, distance_delta_estimate, csi_delta, csi_age]` where
    `relative_velocity_along_link = (p_u−p_v)·(v_u−v_v)/|p_u−p_v|` (signed closing rate: <0 approaching,
    >0 departing; local — u knows v via neighbour broadcast), `distance_delta = rel_vel·dt`, `csi_delta =
    psucc_t − psucc_{t−1}` (env-computed from consecutive frames; 0 at t=0), `csi_age = 0` (every frame
    is freshly measured in this env — an honest placeholder slot for stale-CSI regimes).
- `graph_payload` (the SHARED static-path featurizer) is UNTOUCHED → static path byte-identical; the
  append happens in `dynamic_frames.observation` (dynamic-only).
- CLI `--motion-features` (default off); `dynamic_rl` threads it + records it in `mechanism_activation.json`.
- The actor auto-sizes from the observation dims (no architecture change). The centralized critic reads
  the same per-node features → it gets ALL nodes' velocities (training-only global view); the deployed
  actor only aggregates its physical neighbours (local).

**Controlled variables:** reward/objective (D2), split (D3), evaluator (D4), actor/critic architecture,
decoder, sampler geometry. Only the observation's feature set changes, and only when the flag is on. No
mechanism added. Deployed actor stays local/neighbour-only.

**Dataset:** existing `sample_dynamic_scenes` (single-RSU mobility; D1 replaces geometry later).

**Required failing tests (fail on `715ec19`, pass after):**
- `test_dynamic_actor_observation_contains_velocity` — with `motion_features=True`, nf gains the 5 local
  velocity columns (each row == that node's own velocity); absent when off (byte-identical nf width).
- `test_relative_velocity_feature_changes_sign` — approaching vs departing endpoints give opposite-sign
  `relative_velocity_along_link`.
- `test_memoryless_with_velocity_can_distinguish_approach_vs_depart` — two frames with IDENTICAL current
  CSI but opposite motion produce DIFFERENT observations (so a memoryless actor can tell them apart).
- `test_critic_can_read_training_only_velocity` — the centralized critic forwards a motion-feature obs and
  its input contains all nodes' velocities (training-only global view).
- `test_actor_motion_features_are_local_no_global_fields` — a node's motion features depend ONLY on its
  own (and, for edges, its neighbour's) motion; changing a NON-incident node's velocity leaves them
  unchanged (locality / no global aggregate).

**Expected activation:** `mechanism_activation.json` records `dynamic_task.motion_features=true` and the
node/edge motion feature names.

**Success criterion:** the 5 tests pass; full unit + contract suites green; `--dynamic --motion-features`
smoke exits 0 with the activation flag set and a wider observation; default (no flag) byte-identical.

**Failure criterion:** if the wider observation destabilizes the smoke vs the no-motion baseline, flag it
(do not silently drop features); the A/B (current-CSI vs velocity) is D8, not decided here.
