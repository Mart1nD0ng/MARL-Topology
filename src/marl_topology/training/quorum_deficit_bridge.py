"""Q4 (POMDP-QP-FAR): training-only bridge from a topology to D_quorum on the evaluator's REAL matrices.

The reliability ``C``, energy/latency, and the quorum deficit ``D_quorum`` are all read from the SAME
reference Stage21 evaluator (Contract D5.5 same-evaluator口径): ``C``/energy via ``evaluator.evaluate``
(the real fixed-set robust path the reward uses), ``D_quorum`` via the SAME message matrices the
reliability is built from (``_reliability_inputs``). This is the bridge the Q4 alignment test and (only
if Q4 passes) the Q9 potential use. It calls the CENTRAL evaluator -> TRAINING-ONLY (gate-exempt
``training/``); the deployed actor never calls it.
"""

from __future__ import annotations

from marl_topology.protocol.pbft_reliability import FAULT_FILTER_REMOVE_LARGEST
from marl_topology.protocol.quorum_deficit import expected_initiator_quorum_deficit


def reference_evaluator(evaluator):
    """Unwrap the reference Stage21 evaluator (the vectorized evaluator wraps it as ``._ref``); the
    reference exposes ``_reliability_inputs`` + ``evaluate`` and is float-identical (1e-9) to vectorized."""
    return getattr(evaluator, "_ref", evaluator)


def _clean(selected_edges) -> tuple[str, ...]:
    return tuple(sorted({str(e) for e in selected_edges}))


def topology_quorum_deficit(evaluator, selected_edges, *, cvar_alpha: float = 0.9) -> dict | None:
    """``D_quorum`` for a topology, computed on the evaluator's REAL per-phase message matrices. Returns
    None when there is no fault-tolerant quorum (< 4 validators), matching ``evaluate``'s C=0 branch."""
    ref = reference_evaluator(evaluator)
    ri = ref._reliability_inputs(_clean(selected_edges))
    validators = tuple(ri["validators"])
    if len(validators) < 4:
        return None
    return expected_initiator_quorum_deficit(
        validators, int(ri["fault_tolerance"]),
        pre_prepare_matrix=ri["pre_prepare"], prepare_matrix=ri["prepare"], commit_matrix=ri["commit"],
        fault_filter_mode=FAULT_FILTER_REMOVE_LARGEST, cvar_alpha=cvar_alpha)


def topology_reliability(evaluator, selected_edges) -> dict:
    """True ``C`` / energy / latency for a topology from the reference evaluator (the real path the
    reward uses; fixed_set -> robust worst-case reliability)."""
    ref = reference_evaluator(evaluator)
    m = ref.evaluate(_clean(selected_edges)).metrics
    return {"consensus": float(m["consensus_success_probability"]),
            "energy": float(m.get("energy", 0.0)), "latency": float(m.get("latency", 0.0))}
