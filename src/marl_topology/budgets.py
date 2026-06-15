"""Single source of truth for per-node communication budgets (Route B).

Heterogeneous, kind-aware endpoint (tx/rx) budgets: an RSU base-station can
sustain many more simultaneous links than a vehicle. BOTH the heuristic TEACHER
(feasibility labelling) and the ACTOR (sampler / deployment assembler) use THESE
SAME budgets via the single producer ``node_budgets_for_scene``, so the teacher
never labels a topology feasible that the actor's action pipeline cannot deploy.
Keeping them apart is exactly what pinned tau_feasible at 0 across Stage 33/34
(the teacher needed RSU-degree 5-7 stars; the actor's assembler was capped at
rsu=4 -> the targets were unreachable).

Design (soft, scale-adaptive ceiling -- NOT a tuned degree cap):
- These are GENEROUS, hardware-grounded radio-chain ceilings. An RSU is a
  massive-MIMO base station (many RF chains, here 64); a vehicle OBU has few
  (here 2); a pedestrian UE one. The ceiling is large enough that it almost never
  binds for realistic node counts -- it is a SOFT advisory cap.
- The REAL feasibility gate is psucc >= tau (the env evaluator under shared-
  spectrum interference), which already penalises over-dense self-interfering
  topologies. The candidate degree (at most N-1) and psucc do the actual limiting;
  the ceiling only reflects hardware.
- A per-node interference-DERIVED cap was deliberately rejected: it would double-
  model the interference the env already evaluates, and an SINR sweep over a star
  is orientation-degenerate (a hub is the sole transmitter of its incident links,
  so they do not mutually interfere -> the sweep yields full degree).
- Why this matters at scale: the unique feasible consensus structure is a dense
  RSU-centred star of degree N-1. The OLD rsu=8 was sized so that star fit only for
  node counts <= 9; it provably rejected the feasible star at N >= 11 (degree
  N-1 > 8) -> tau_feasible=0 at scale. The generous ceiling removes that wall.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

# kind (NodeKind.value) -> generous hardware radio-chain ceiling (soft advisory
# cap; psucc >= tau is the real feasibility gate -- see module docstring). RSU/base
# = massive-MIMO chain count; vehicle/pedestrian = few-antenna UE. Sized large so it
# never artificially blocks the feasible RSU star at scale (the old rsu=8 rejected
# the degree-(N-1) star for N >= 11).
CANONICAL_ENDPOINT_BUDGET_BY_KIND: dict[str, int] = {
    "rsu": 64,
    "base_station": 64,
    "vehicle": 2,
    "pedestrian": 1,
}
_DEFAULT_BUDGET = 2
EDGE_ENDPOINT_SEPARATOR = "--"


def budget_for_kind(kind: str) -> int:
    return int(CANONICAL_ENDPOINT_BUDGET_BY_KIND.get(kind, _DEFAULT_BUDGET))


def node_budgets_for_scene(scene) -> tuple[tuple[str, int], ...]:
    """Per-node (node_id, budget) tuple derived from each node's kind.

    The single helper both the teacher (feasibility filter) and the actor
    (sampler endpoint_budgets) use, so they are guaranteed consistent.
    """

    return tuple(
        (str(node.node_id), budget_for_kind(node.kind.value)) for node in scene.nodes
    )


def _endpoints(edge_id: str) -> tuple[str, str]:
    parts = str(edge_id).split(EDGE_ENDPOINT_SEPARATOR)
    if len(parts) != 2:
        raise ValueError(f"unexpected edge_id format (expected 'a--b'): {edge_id!r}")
    return parts[0], parts[1]


def is_budget_feasible(
    edges: Iterable[str],
    node_budgets: Sequence[tuple[str, int]] | Mapping[str, int],
) -> bool:
    """True iff every node's incident degree in ``edges`` is within its budget.

    Symmetric, undirected accounting (debit BOTH endpoints of an a--b edge),
    mirroring the actor's BudgetAwareSequentialProposalSampler._eligible and the
    deployment assembler (DEFAULT_TX == DEFAULT_RX per kind, so a single symmetric
    rule is faithful and avoids a directed-orientation hazard). Raises loudly on a
    node_id/endpoint key mismatch rather than silently passing.
    """

    budgets = dict(node_budgets)
    degree: dict[str, int] = {}
    for edge_id in edges:
        for endpoint in _endpoints(edge_id):
            if endpoint not in budgets:
                raise KeyError(
                    f"edge endpoint {endpoint!r} has no node budget "
                    f"(node_id/edge-endpoint key mismatch); budget keys={sorted(budgets)}"
                )
            degree[endpoint] = degree.get(endpoint, 0) + 1
            if degree[endpoint] > budgets[endpoint]:
                return False
    return True
