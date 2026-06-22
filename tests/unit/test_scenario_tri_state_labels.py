"""Recalibration step 2c: the scenario generator records the honest tri-state solvability.

A finite-search MISS is ``unknown`` (a feasible topology may still exist), NEVER
``certified_infeasible`` (hard-constraint #11). The binary ``feasible_exists`` stays, and
the new ``solvability_status`` is recorded beside it.
"""

from __future__ import annotations

from marl_topology.data.stage31_scenario_generator import (
    ProductionScenarioConfig,
    generate_production_scenarios,
)
from marl_topology.solvability import CERTIFIED_INFEASIBLE, UNKNOWN, WITNESS_FEASIBLE


def test_specs_carry_tri_state_with_no_false_infeasible() -> None:
    # node counts that yield a mix of feasible + search-miss scenes.
    specs = generate_production_scenarios(
        ProductionScenarioConfig(seed=31, scenario_count=12, node_count_choices=(5, 6, 7))
    )
    statuses = {spec.solvability_status for spec in specs}
    # a finite search can never certify infeasibility (#11) -> only W or U appear.
    assert CERTIFIED_INFEASIBLE not in statuses
    assert statuses <= {WITNESS_FEASIBLE, UNKNOWN}
    for spec in specs:
        # consistent with the legacy binary: feasible_exists iff a witness was found.
        assert (spec.solvability_status == WITNESS_FEASIBLE) == spec.feasible_exists
    # the generator targets infeasible-FAMILY scenes (nodes pushed out of range); under the
    # corrected semantics those are `unknown` (search miss), never certified infeasible.
    assert UNKNOWN in statuses
