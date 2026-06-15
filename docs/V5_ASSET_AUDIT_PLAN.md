# V5 Asset Audit Plan

## Controlled Object

The controlled object is the candidate set of ideas, scripts, metrics, skills, and configurations in `D:\PhD_works\v5`.

## Boundary

- Read only: `D:\PhD_works\v5`.
- Write only audit outputs in this repository, under `harness/reports/` or `docs/MIGRATION_LEDGER.md`.
- Do not run training or legacy phase scripts during audit.

## Procedure

1. Inventory candidate files by purpose, not by phase number.
2. Map each candidate to a new contract: metric, physics, protocol, reward, Dec-POMDP, topology oracle, or training.
3. Record evidence, risks, required tests, and owner decision in the migration ledger.
4. Reject or defer any candidate whose semantics are ambiguous.
5. Prefer wrappers or rewrites over direct copy.

## Sensors

- File inventory.
- Grep results for metric names and reward terms.
- Contract test requirements.
- Entropy audit notes.
- Ledger status.

## Acceptance

- No candidate is migrated during audit.
- Old reward remains reject / legacy-only unless a future task explicitly opens an ablation.
- Old phase scripts remain legacy-only.

