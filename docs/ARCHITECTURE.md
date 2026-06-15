# Architecture

## Clean-Core Principle

The repository starts from contracts, interfaces, and tests. Legacy lessons from `v5` can inform a clean skeleton layer only through the learning ledger and contract tests.

## Package Boundaries

| Package | Responsibility | Current Status |
|---|---|---|
| `geometry3d` | 3D points, boxes, roads, buildings, and ray queries | Skeleton only |
| `scenario` | Scenario generation and static scene definitions | Skeleton only |
| `channel` | Path loss, shadowing, SINR, and interference models | Skeleton only |
| `link` | Link success, latency, and energy interfaces | Skeleton only |
| `protocol` | PBFT quorum and consensus reliability contracts | Skeleton only |
| `objectives` | Reward and constraint definitions | Skeleton only |
| `metrics` | Metric aggregation and CSV schema | Skeleton only |
| `env` | MARL environment interfaces | Skeleton only |
| `topology` | Topology proposal, oracle, and feasibility checks | Skeleton only |
| `policies` | Actor/critic interfaces and observation boundaries | Skeleton only |
| `data` | Replay, datasets, and metric snapshots | Skeleton only |
| `models` | Future network modules after architecture review | Skeleton only |
| `training` | Future training orchestration after contracts | Skeleton only |
| `evaluation` | Hard evaluation, baselines, and oracle comparisons | Skeleton only |

## Dependency Direction

Contracts should drive code:

1. Docs define semantics.
2. Config schemas encode allowed modes.
3. Source modules expose narrow interfaces.
4. Contract tests guard each interface.
5. Harness tasks audit future changes.

## Non-Goals In This Scaffold

- No migrated legacy model code.
- No full 3D physics simulator.
- No training loop.
- No GPU or deep-learning dependencies.
