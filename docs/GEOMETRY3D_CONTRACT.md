# Geometry3D Contract

## Responsibility

Represent deterministic 3D urban geometry and compute visibility facts needed by
communication simulation.

## Inputs

- `scenario_id`
- `BuildingBox` records with 3D footprint, height, and optional material tag
- road and lane records with 3D centerline or surface references
- vehicle records with 3D position in meters
- RSU and base-station records with 3D position in meters
- transmitter id and receiver id for a queried link

## Outputs

- `distance_3d_m`
- `los_state`: `los`, `nlos`, or declared extension value
- `blocker_records`: ordered building or object intersections
- `ray_segment_m`: transmitter-to-receiver segment in meters
- optional `visibility_regime`

Stage 3.1 implements ray-box intersection as the deterministic visibility
sensor.

These outputs are geometry records. They are not link success, network delivery,
or application consensus metrics.

Geometry visibility records are not link success, not network delivery, and not
application consensus. They are not application consensus.

## Units

- positions: meters
- building dimensions: meters
- distance: meters
- heights: meters
- angles, if later added: radians unless declared otherwise

## Assumptions

- Stage 3.1 starts with axis-aligned building boxes unless a later contract
  admits mesh geometry.
- Ray-box intersection is deterministic.
- LoS is clear when the ray segment has no blocking building intersection.
- NLoS is assigned when at least one declared blocker intersects the ray segment.
- Urban gaps and corridors are geometry facts, not channel shortcuts.

## Omitted Components

- diffraction;
- reflection and multipath;
- material penetration loss;
- moving blockers;
- curved earth or terrain;
- probabilistic map uncertainty;
- rendering.

## Tests

- 3D distance is correct and nonnegative.
- Building blocker is recorded when a box intersects the ray.
- Changing building height can change LoS/NLoS.
- Urban gap fixture preserves a clear corridor.
- RSU height fixture can restore LoS when the ray clears the blocker.
- Blocker ids and intersection ordering are stable.

## Stage 3.1 Implementation Status

Implemented interfaces:

- `Point3D`
- `BuildingBox`
- `RoadSegment`
- `Lane`
- `LoSState`
- `BlockerRecord`
- `VisibilityRecord`
- `ray_box_intersection`
- `evaluate_visibility`

Active visibility regime:

`stage3_axis_aligned_boxes`

Implemented fixtures:

- `free_space_close`
- `free_space_far`
- `blocked_by_building`
- `urban_gap_los`
- `urban_canyon_nlos`
- `rsu_high_los`

Deferred from Stage 3.1:

- channel path loss;
- shadowing;
- SINR;
- interference;
- packet success probability;
- link transmission latency and energy;
- network delivery;
- PBFT/application consensus.

## Failure Modes

- 2D shortcuts hidden behind a 3D interface.
- Inclusive/exclusive box boundary ambiguity.
- LoS state changes with node ordering.
- Blocker records missing scenario or building ids.
- Building gaps accidentally filled by footprint overlap.

## GOAL_SKELETON Coupling

This contract hardens `Scene3D`, `CandidateGraph`, and the geometry input side of
`LinkModel`.

## Deferred To Stage 4

No PBFT quorum success, consensus reliability, consensus reward, or
application-level deadline objective is defined here.
