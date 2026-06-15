# V5 Learning Strategy

## Position

MARL-Topology inherits the goal from `v5`, not the structure.

Goal inherited:

- Learn topology planning over vehicles and RSUs as graph nodes.
- Preserve Dec-POMDP / MARL properties.
- Make consensus reliability satisfy a constraint.
- Reduce latency and energy after reliability is feasible.

Structure reset:

- Clean skeleton first.
- Contracts before implementation.
- Shared modules before scripts.
- Metric governance before metric proliferation.
- Baselines before complex MARL architectures.

## What V5 Is

`D:\PhD_works\v5` is a read-only experience library:

- Project goal history.
- Simulation design attempts.
- Metric and reward lessons.
- Failure records.
- Useful edge cases.
- Negative examples of semantic debt.

## What V5 Is Not

`v5` is not:

- A code template.
- A default module source.
- A phase-script source for the new mainline.
- A reward source.
- A metric naming source.
- An actor/critic architecture default.

## When To Consult V5

Consult `v5` only when a clean skeleton layer has a concrete design decision:

- `Scene3D`: scenario representation alternatives.
- `CandidateGraph`: node and edge candidate construction.
- `LinkModel`: link abstraction and sanity cases.
- `TopologyEvaluator`: how topology choices affect communication.
- `ConsensusSuccess`: protocol edge cases.
- `TopologyOracle`: feasibility and counterfactual checks.
- `PolicyBaselines`: baseline behavior and leakage risks.

Do not browse v5 just to find code to copy.

## How To Learn From V5

For each lesson:

1. State the clean skeleton layer.
2. State the design question.
3. Record the v5 observation.
4. Classify it as useful route, caution, or counterexample.
5. Define the new project contract or test it affects.
6. Keep implementation in the new clean-core style.

## Explicit Avoidance Rules

- Do not migrate actor/critic/COMA code in Stage 0.
- Do not copy old reward.
- Do not copy old phase scripts as the new mainline.
- Do not import old metric names without metric registration.
- Do not treat old training outcomes as proof without reproducible evidence.

## Acceptance

- Learning audit outputs update `docs/MIGRATION_LEDGER.md` as a learning ledger.
- Every adopted lesson maps to a clean skeleton layer and test.
- Failed v5 routes are recorded as counterexamples, not silently repeated.

