# D5 — decision

**Result: KEEP.** Local motion features are added to the dynamic actor observation (opt-in), so a
memoryless actor now has the information to predict next-frame link evolution — a precondition for a
fair Markovness test (D8). Adversarial verification PASS (no residual gaps).

## What changed (one variable: the observation feature set, opt-in)
- `DynamicScene` carries per-node `velocities` (from the `NodeMotion`s), `dt_s`, and a `motion_features`
  flag. `observation(t)` appends, ONLY when on:
  - node: `[velocity_x, velocity_y, speed, heading_sin, heading_cos]` (each node's OWN velocity — local).
  - edge `(u,v)`: `[relative_velocity_along_link, distance_delta, csi_delta, csi_age]` —
    `rel_vel = (p_u−p_v)·(v_u−v_v)/|p_u−p_v|` (signed closing rate: <0 approaching, >0 departing; local
    via neighbour broadcast), `distance_delta = rel_vel·dt`, `csi_delta = psucc_t − psucc_{t−1}`
    (env-computed from consecutive frames, 0 at t=0; locally measurable in deployment), `csi_age = 0`
    (every frame freshly measured here — an honest placeholder slot for stale-CSI regimes).
- `graph_payload` (the SHARED static-path featurizer) is UNTOUCHED → the static T=1 path is byte-identical;
  the append is in `dynamic_frames.observation` (the single dynamic observation builder).
- CLI `--motion-features` (default off); `dynamic_rl` threads it and records `motion_features` +
  the feature names in `mechanism_activation.json`. The actor auto-sizes from the observation dims (no
  architecture change). The centralized critic reads the same per-node features → it gets ALL nodes'
  velocities (training-only global view); the deployed actor only aggregates physical neighbours (local).

## Tests (failing-first; fail on `715ec19`, pass after)
- `test_dynamic_actor_observation_contains_velocity` (nf +5 / ef +4; veh_0 row == its own velocity),
  `test_relative_velocity_feature_changes_sign` (approach<0<depart),
  `test_memoryless_with_velocity_can_distinguish_approach_vs_depart` (same CSI, opposite motion → different obs),
  `test_critic_can_read_training_only_velocity`, `test_actor_motion_features_are_local_no_global_fields`.
- Unit suite **671/0**; contract **63/0**; `--dynamic --motion-features` smoke **exit 0** (activation flag
  + feature names set, observation wider); default off → byte-identical.

## Adversarial verification (focused single agent — proportional to a contained, well-tested change)
PASS on all 4 lenses, no residual gaps:
1. **Decentralization/locality:** all features local (own velocity + incident-pair quantities); no global
   aggregate; csi_delta is psucc_t − psucc_{t−1} for the same edge (no future/teacher leakage).
2. **Correctness:** rel_vel sign right; distance_delta/heading/csi_delta consistent; node/edge feature
   order aligns with `graph.node_ids` / `graph.edges`.
3. **Byte-identity off + no bypass:** default off everywhere; `graph_payload` untouched; the observation
   builder is unique (the D4 lesson applied — no vectorized/alternate obs path bypasses it); actor
   auto-sizes.
4. **Test adequacy:** 5 genuine anti-pseudo-tests; fail on pre-D5.

## Scope / next
- D5 enables the Markovness test; it does NOT itself claim recurrence is useless — the
  current-CSI/velocity/recurrent/velocity+recurrent A/B is **D8** (after D6/D7). The myopic reference is
  motion-agnostic by design (a candidate-enumerating reference, not a learned actor).
- Next: **D6** (warm-start protection: decoder-aware / BCSP-subset-likelihood imitation + teacher-KL/BC
  anchor so PPO does not destroy the feasible warm-start; report teacher source / leakage /
  warm-start-alone return / post-RL drift).
