---
name: marl-dec-pomdp-design
description: Design actor/critic information boundaries for Dec-POMDP-compliant MARL-Topology policies.
---

# MARL Dec-POMDP Design

## When To Use

- Before actor architecture, replay format, observation schema, or checkpoint-loading changes.
- When centralized critic features are introduced.

## When Not To Use

- For pure documentation spelling fixes.
- To justify global actor inputs during deployment.

## Procedure

1. Read `docs/DEC_POMDP_CONTRACT.md`.
2. List actor allowed and forbidden fields.
3. List critic-only centralized fields.
4. Define deployment actor schema.
5. Add leakage tests and negative tests.

## Output Format

```text
Actor schema:
Allowed local fields:
Forbidden global fields:
History source:
Critic-only fields:
Deployment boundary:
Leakage tests:
V5 inheritance check:
Forbidden defaults avoided:
Residual risks:
```

## Quality Gates

- Actor can run without critic-only tensors.
- Dataset and replay columns are audited.
- Negative test fails on forbidden global actor field.
- Actor schema is local-first before architecture choices are discussed.

## V5 Anti-Inheritance Calibration

- Read `docs/SKILL_CALIBRATION.md` and lesson `L009_dec_pomdp_boundary_before_actor`.
- Do not treat v5 full graph tensors, topology history, oracle labels, or centralized labels as deployment actor defaults.
- LSTM, GNN, GNN+LSTM, COMA, and direct edge-delta critics remain future options until actor-local observation and leakage tests exist.
- Any memory feature must state whether it is local history, permitted message history, or critic-only training data.

## Failure Modes

- Global topology hidden in an actor feature.
- Future state included through replay labels.
- Checkpoint loader silently couples actor and critic features.
