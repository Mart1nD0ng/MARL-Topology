# Run Manifest Artifact Contract

## Current Status

The active run-manifest and artifact contract is design-only for writers. It
now also has a dry-run exit-gate validator. Artifact writing remains blocked.

Authoritative Stage 5.9 document:

- `docs/STAGE5_9_TRAINING_RUN_MANIFEST_ARTIFACT_CONTRACT.md`

Authoritative Stage 5.10 exit-gate document:

- `docs/STAGE5_10_RUN_MANIFEST_VALIDATOR.md`

Structured manifest:

- `src/marl_topology/training/run_manifest_contract.py`
- `src/marl_topology/training/run_manifest_validator.py`

## Execution Boundary

Stage 5.9 does not write artifacts, implement manifest writers, create
checkpoints, export datasets, run training, add model code, calibrate reward
weights, select final `tau_consensus`, or migrate v5 code.

Stage 5.10 adds a dry-run validator only. It accepts or rejects in-memory
manifest dictionaries and artifact-root strings. It does not create directories,
write manifests, write artifacts, export datasets, create checkpoints, run
training, add model code, calibrate reward weights, select final
`tau_consensus`, or migrate v5 code.

## Artifact Root

Future artifacts must remain under:

```text
result_save
```

The scaffold baseline is `.gitkeep` only.

The Stage 5.10 validator rejects artifact roots that escape this directory and
rejects legacy reference paths as artifact roots.

## Future Manifest Fields

Future run manifests must include owner approval id, stage id, config id,
scenario set id, split id, seed, seed group id, code version marker, contract
ids, metric registry version, physics regime id, protocol model id, objective
contract id, surrogate config id, normalization reference id, architecture
contract id, replay schema version, and artifact policy id.

## Future Required Work

Stage 5.10 now provides the manifest completeness validator, artifact path
validator, result root containment check, and dry-run report.

Future work before any artifact write is allowed:

- run id collision check;
- retention-budget check;
- owner-approved writer implementation;
- explicit execution authorization.
