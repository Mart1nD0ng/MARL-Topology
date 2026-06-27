# D1 — decision

**Result: KEEP (the founding gap #1 is closed — the dynamic data can now genuinely be a 4-RSU urban
grid).** The DATA infrastructure is delivered, opt-in, verified; the urban A/B headline is D13.

## What changed (one variable: the dynamic data source, opt-in)
- New `dynamic_frames.sample_dynamic_urban_scenes`: each scene is built via `build_urban_grid_scene`
  with `UrbanGridConfig(rsu_count=4 default, ...)` → exactly `rsu_count` RSUs at distinct intersections,
  `G×G` building blocks (real NLOS canyons), and a street grid (roads + lanes). Vehicles drive ALONG the
  streets via `_road_constrained_motions` (axis-aligned grid-street CONSTANT velocity — exactly one of
  vx/vy is 0; genuinely different from the free-heading `_sample_vehicle_motions`; honestly described, NO
  lane-change/turn dynamics claimed). RSUs are fixed. Advanced via the existing `dynamic_scene_from_motion`
  (so the node-id + candidate-edge-id sets are invariant across frames, frame 0 = the static urban scene,
  and each frame is the real Stage-21 measurement on the moved geometry).
- `dynamic_urban_manifest`: provenance computed FROM the actual scenes (rsu_count, urban_grid_config,
  building/road/lane counts, mobility_model, speed_distribution, content_hash) — so the data description
  is verifiable against the source (Contract §4.2; no single-RSU-written-as-urban).
- `run_dynamic_training`: `--dyn-data {random,urban}` dispatch (default `random` →
  `sample_dynamic_scenes`, byte-identical); writes `dynamic_data_manifest.json` and records
  `data_source` + `urban_rsu_count` in the activation when urban. The old single-RSU
  `sample_dynamic_scenes` is kept as the `dynamic_random_geometry` ablation.

## Tests (failing-first; fail on `a4bb937` with ImportError, pass after)
- `test_dynamic_urban_has_four_rsus` (exactly 4 RSUs rsu_0..rsu_3), `test_dynamic_urban_uses_grid_and_buildings`
  (G×G buildings + street grid), `test_vehicle_motion_is_road_constrained` (axis-aligned, exactly one
  velocity component 0), `test_edge_ids_stable_across_frames`, `test_frame0_matches_static_urban_context`,
  `test_channel_changes_with_motion` (the channel evolves with motion).
- Unit suite **703/0**; contract **63/0**; `--dyn-data urban` smoke exit 0 (manifest: sampler=
  sample_dynamic_urban_scenes, rsu=4, buildings=9, road-constrained mobility, content hash).

## Adversarial verification (focused single agent, 4 claims, PASS, no blockers)
1. **Genuinely 4-RSU urban**: real `build_urban_grid_scene` with 4 RSU + G×G buildings + street grid; no
   silent fallback to 1 RSU / no buildings.
2. **Road-constrained motion**: axis-aligned (one velocity component 0), determined by the vehicle's
   street; honestly described (constant-velocity, no turn dynamics); different from free-heading random.
3. **Manifest matches source + invariants**: real config + content hash from the scenes; ids stable,
   frame-0 static, channel evolves.
4. **Byte-identity off + no over-claim**: default random → `sample_dynamic_scenes` (byte-identical); urban
   opt-in; activation honest; D1 is DATA infrastructure (NO urban RESULT claimed); the urban smoke
   feasibility 0.0 (urban NLOS is genuinely harder) is reported, not hidden. One nit fixed (docstring
   "4-RSU" → "rsu_count default 4").

## Scope / next — ALL 10 GAPS NOW CLOSED
- D1 closes gap #1 (the campaign's founding gap: single-RSU random geometry mislabeled as 4-RSU urban).
  **All 10 gaps from `CURRENT_DYNAMIC_REPAIR_STATUS.md` are now resolved.** D1 delivers the urban DATA; it
  makes NO urban performance claim — the urban headline (and the `dynamic_random_geometry` vs
  `dynamic_urban_4rsu` contrast) is the D13 campaign.
- Next: **D13** — the full multi-seed campaign on the urban data with all the D9–D12 mechanism A/Bs
  (COMA credit, SCQ held-Q fidelity, chance residual / CVaR / Pareto hypervolume, PNA vs MLP), per-seed +
  CI, scope + failure modes. Then **D14** (docs/report/README reconciliation).
