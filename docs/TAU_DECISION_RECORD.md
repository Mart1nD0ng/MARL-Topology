# Tau Decision Record — `tau_requirement_min = 0.9`

**Status:** Accepted (consolidation of an existing owner decision; no value change).
**Owner approval id:** `owner_approved_stage31_production_readiness_unfreeze`
**Scope:** the consensus-feasibility constraint `consensus_success_probability >= tau`
used across Stages 21 / 31 / 32 / 33 and the learning-evidence completion gate.

```
CANONICAL_TAU_REQUIREMENT_MIN = 0.9
```

This record is the single source of truth for the `tau_requirement_min` constant.
It exists because the value was previously asserted as a bare literal in **ten**
independent modules with no shared, cited basis, and because the rationale for 0.9
was scattered across several docs. `tests/contract/test_tau_decision_record.py`
pins every in-source definition to the value declared here so the number can never
drift unnoticed.

---

## Decision

The minimum acceptable end-to-end PBFT **consensus** success probability for a
planned communication topology to count as *feasible* is **0.9**. A topology whose
`consensus_success_probability` is below 0.9 is treated as infeasible.

This is a **requirement floor**, not a tuned hyper-parameter and not a calibrated
optimum. The engineering job is to make 0.9 *reachable* by good topologies, not to
lower it. Lowering 0.9 requires a new owner decision; the existing RAISE-on-change
guards in code enforce this (e.g. `stage31_scenario_generator.py`,
`stage21_objective_stack_evidence.py`, `stage33_graph_structure_dataset.py`,
`requirement_feasibility_diagnosis.py`).

## Basis

1. **Owner risk-class floor.** Recorded in `docs/STAGE31_OWNER_DECISION_AND_UNFREEZE.md`:
   "a planned topology whose consensus success probability is below 0.9 has no
   practical value, so the constraint is a real requirement, not a tunable." This is
   the primary basis and carries the approval id above.

2. **Per-link reliability anchor (URLLC-style).** The link layer already targets a
   per-link delivery reliability of **0.99** (`stage31_scenario_generator.py`
   `RELIABLE_LINK_PROBABILITY = 0.99`; `PhysicsRegime.target_reliability = 0.99`).
   PBFT consensus aggregates many such links through three phases with quorum and
   fault-tolerance amplification, which pulls the *end-to-end* consensus-commit
   probability **below** the per-link 0.99. A consensus-level requirement of 0.9
   sitting below the 0.99 per-link target is therefore internally consistent: 0.9 is
   the accepted *end-to-end* number after PBFT amplification, not the per-hop number.
   (Note: 0.9 is **not** a radio-layer per-packet reliability — citing "URLLC = five
   nines" as the basis for 0.9 would be a layer category error.)

3. **Evidence that 0.9 is a meaningful constraint (non-saturation).** On the Stage 5.0h
   alpha suite, tau = 0.9 produces a genuine non-saturated split (6/24 feasible across
   8 families) — neither trivially passed nor impossible — which is exactly what a
   useful constraint should do. Stress checks at 0.95 / 0.99 still leave some feasible
   rows, so 0.9 is not arbitrarily lax. See
   `docs/STAGE5_0H_REQUIREMENT_ANCHORED_FEASIBILITY_DIAGNOSIS.md` and the procedural
   generator gradient tests (`tests/unit/test_procedural_generator.py`).

## Relationship to the formal `tau_consensus`

There are two distinct tau tracks, and they are deliberately reconciled here:

- **`tau_requirement_min = 0.9` (this record):** the operational requirement floor,
  decided and frozen.
- **`tau_consensus` (formal objective-contract parameter):** intentionally left
  *open* for final calibration (`docs/OBJECTIVE_CONTRACT.md`,
  `docs/REWARD_CONTRACT.md`; `evaluation/tau_consensus_calibration_report.py` refuses
  to auto-select it).

0.9 is the **lower bound** for `tau_consensus`: the formal calibrated value (which a
future owner decision may set higher) is bounded below and decided at 0.9, even
though the upper-bound calibration remains open.

## Open item (honest limitation)

0.9 is a defensible *floor* derived from owner intent plus the per-link/end-to-end
consistency argument, but it has **not** been derived from a quantified tolerable
consensus-failure budget (e.g. an acceptable consensus-failure rate per V2X decision
round) nor compared against a deployment SLA. Upgrading the basis from "floor" to
"calibrated risk class" is deferred to the `tau_consensus` calibration track and is
not required for the value to stand as a requirement floor.
