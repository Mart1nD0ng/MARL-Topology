# MARL-Topology

Clean-core scaffold for a 3D V2X / PBFT / MARL topology-control simulation project.

The project goal is to redesign the v5 target with a lower-entropy structure. `v5` is treated as a read-only experience library, not a code template. The first-stage repository contains contracts, harness, skills, and a Python package skeleton only. It does not contain migrated model code, training code, or a 3D physics simulator implementation.

## Scope

- 3D city topology with buildings, road geometry, vehicles, RSUs, and base stations.
- Link reliability, latency, energy, and consensus-success metric governance.
- Reward framing: consensus reliability is a constraint; latency and energy are objectives.
- MARL / Dec-POMDP discipline: deployment actors use local observations and history only.
- Centralized critics are allowed for training if deployment information boundaries are preserved.

## Current Status

This is a controlled learning scaffold:

- Contracts live in `docs/`.
- Skills live in `.agents/skills/`.
- Harness rubrics, tasks, templates, and scripts live in `harness/`.
- Source modules are empty package boundaries under `src/marl_topology/`.
- Tests currently verify scaffold and contract file presence.

## Verification

```powershell
python -m pytest -q
python harness\scripts\validate_tasks.py
python harness\scripts\score_rubric.py docs\CONTROL_MODEL.md harness\rubrics\cybernetic_engineering_rubric.yaml --out harness\reports\control_model.score.json
```

## Boundary Rules

- `D:\PhD_works\v5` is read-only legacy reference.
- The project inherits the goal from v5, not its structure.
- No old reward, model, or phase script is migrated by default.
- No training should run until reward, metric, and evaluation contracts are explicit.
