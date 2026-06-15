"""Phase 2 adaptive budget: the per-node endpoint budget is a GENEROUS, scale-safe
hardware ceiling (psucc>=tau is the real feasibility gate), not a tuned degree cap.

The old fixed rsu=8 provably rejected the only feasible structure -- the RSU-centred
star of degree N-1 -- once N>=11 (degree 9+ > 8), pinning tau_feasible at 0 at scale.
These tests pin the fix: the budget admits the full RSU star at large N, a vehicle
stays hardware-limited, and the single-producer consistency holds.
"""

from marl_topology.budgets import (
    CANONICAL_ENDPOINT_BUDGET_BY_KIND,
    budget_for_kind,
    is_budget_feasible,
)


def _star_edges(hub: str, spokes: list[str]) -> list[str]:
    return ["--".join(sorted((hub, s))) for s in spokes]


def test_rsu_ceiling_admits_full_star_at_large_n() -> None:
    # An RSU star with N=15 (hub degree 14). The generous ceiling admits it (then
    # psucc>=tau decides feasibility); the OLD rsu=8 cap rejected it outright.
    hub = "rsu0"
    spokes = [f"veh{i:02d}" for i in range(14)]
    edges = _star_edges(hub, spokes)
    node_budgets = [(hub, budget_for_kind("rsu"))] + [
        (s, budget_for_kind("vehicle")) for s in spokes
    ]
    assert is_budget_feasible(edges, node_budgets) is True
    # explicit regression of the documented scale bug: degree-14 hub fails under rsu=8.
    old_budgets = [(hub, 8)] + [(s, 2) for s in spokes]
    assert is_budget_feasible(edges, old_budgets) is False


def test_rsu_ceiling_is_generous_and_hardware_grounded() -> None:
    # must be large enough to never artificially block a realistic deployment's
    # feasible star (the scale bug was rsu=8 binding at N>=11).
    assert CANONICAL_ENDPOINT_BUDGET_BY_KIND["rsu"] >= 32
    assert CANONICAL_ENDPOINT_BUDGET_BY_KIND["base_station"] >= 32


def test_vehicle_stays_small_hardware_limited() -> None:
    vb = budget_for_kind("vehicle")
    assert 1 <= vb <= 4  # small UE radio-chain ceiling
    assert budget_for_kind("vehicle") < budget_for_kind("rsu")
    assert budget_for_kind("pedestrian") == 1
    # a vehicle wired to vb+1 hubs exceeds its ceiling -> rejected (uplink is
    # interference-limited; the cap reflects hardware, psucc is the real arbiter).
    veh = "veh0"
    hubs = [f"rsu{i}" for i in range(vb + 1)]
    edges = ["--".join(sorted((veh, h))) for h in hubs]
    node_budgets = [(veh, vb)] + [(h, budget_for_kind("rsu")) for h in hubs]
    assert is_budget_feasible(edges, node_budgets) is False


def test_unknown_kind_defaults_to_small_budget() -> None:
    # an unmapped kind must not silently get the huge RSU ceiling.
    assert budget_for_kind("drone_not_a_real_kind") <= 4
