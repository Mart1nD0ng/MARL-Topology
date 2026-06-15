---
name: marl-topology-oracle
description: Design topology feasibility oracles and counterfactual checks for V2X/PBFT topology control.
---

# MARL Topology Oracle

## When To Use

- When designing feasibility audits, pruning checks, counterfactual topology evaluation, or oracle baselines.
- Before redesigning v5 phase34-style topology audits.

## When Not To Use

- To give actor policies oracle labels at deployment.
- To optimize reward directly without metric contract.

## Procedure

1. Define the topology decision and controlled graph.
2. Define feasible and infeasible reference cases.
3. Connect oracle decisions to protocol and physics contracts.
4. Specify replay/counterfactual rules that avoid actor leakage.
5. Add regression cases.

## Output Format

```text
Topology object:
Oracle purpose:
Inputs:
Outputs:
Feasible cases:
Infeasible cases:
Counterfactual rules:
Tests:
V5 inheritance check:
Forbidden defaults avoided:
Leakage risks:
```

## Quality Gates

- Oracle labels are evaluation or training-only as declared.
- Feasible/infeasible cases are reproducible.
- Counterfactual replay holds scenario state constant.
- Full-mask is only one baseline unless oracle evidence proves stronger claims.

## V5 Anti-Inheritance Calibration

- Read `docs/SKILL_CALIBRATION.md` and lessons `L003_full_mask_not_oracle` and `L005_oracle_before_infeasible_claims`.
- Do not copy v5 phase34/phase36 scripts as oracle implementation.
- Preserve `feasible`, `infeasible`, and `unresolved` as distinct audit outcomes; policy failure alone is not infeasibility.
- Oracle labels must not enter deployment actor observations or action selection.

## Failure Modes

- Oracle output becomes actor input.
- Feasibility ignores PBFT quorum or deadline.
- Phase-script logic is copied instead of redesigned as a library.
