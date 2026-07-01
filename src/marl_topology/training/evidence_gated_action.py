"""R6 (Belief-Guided Residual PPO): the EVIDENCE-GATED residual action (DEPLOYABLE, 0 evaluator at decision).

R4 proved a beneficial-edit DIRECTION signal exists; R5 proved the LOCAL-feature heads can RANK it. R6 turns
that ranking into a deployable DECISION: starting from the ``local_hysteresis`` anchor (0 eval), enumerate the
LOCAL candidate edits (add = budget-feasible non-anchor incident edges, remove = anchor edges), score them with
the FROZEN R5 heads (``edit_scores`` -> edit_logit / repair_pred / safety_pred, local features only), and apply
ONLY the gate-passing edits via the budget-safe mutual decode. Zero gated candidates -> the anchor EXACTLY.

The gate is the single new mechanism (vs R3's ungated all-candidate Bernoulli, which collapsed to the anchor):
  * add-candidate passes iff  repair_pred >= tau_repair  AND  sigmoid(edit_logit) >= tau_edit
  * remove-candidate passes iff safety_pred <= tau_safety AND  sigmoid(edit_logit) >= tau_edit
The decision path is fully decentralized: local features + mutual decode, NO evaluator / true-CSI / critic /
central decoder. The true PBFT C/E/L/J are computed by the evaluator ONLY on the produced topology (current
real channel + closed-form quorum tail) -- never in this decision path. This promotes the R5 heads from
ACTIVE_IN_EVAL to ACTIVE_IN_DEPLOY.
"""

from __future__ import annotations

import torch

from marl_topology.budgets import node_budgets_for_scene
from marl_topology.training.dynamic_baselines import local_hysteresis_action
from marl_topology.training.dynamic_rl import _budgets_edges, _standardize
from marl_topology.training.residual_action import residual_decode_from_flips
from marl_topology.training.residual_repair import _degrees, _ends

_KT, _AT = 0.4, 0.6                      # anchor (local_hysteresis) keep/add thresholds -- identical to R5


def _anchor(obs, prev):
    budgets, edges = _budgets_edges(obs["context"])
    return local_hysteresis_action(obs["ef"], obs["edge_ids"], edges, budgets, prev,
                                   keep_threshold=_KT, add_threshold=_AT)


def local_candidates(anchor, edge_ids, ends, budgets):
    """LOCAL (deployable, 0 eval) candidate enumeration around the anchor. Returns ``(add_ids, remove_ids)``:
    add = budget-feasible non-anchor edges (degree < budget on BOTH endpoints), remove = anchor edges. Depends
    only on anchor membership + local degree/budget (the R5 ``build_edit_examples`` enumeration MINUS the
    evaluator-label computation)."""
    anchor_set = set(anchor)
    deg = _degrees(anchor, ends)
    add_ids, remove_ids = [], []
    for e in edge_ids:
        if e in anchor_set:
            remove_ids.append(e)
        else:
            u, v = ends[e]
            if deg.get(u, 0) < budgets.get(u, 0) and deg.get(v, 0) < budgets.get(v, 0):
                add_ids.append(e)
    return add_ids, remove_ids


def evidence_gated_residual(obs, actor, prev_topology, mean, std, *, tau_edit=0.5, tau_repair=0.0,
                            tau_safety=0.0, mode="full"):
    """Deployable evidence-gated residual action -> dict with the produced topology and decision diagnostics.

    The anchor is directly computed (0 eval). The FROZEN heads (``actor.edit_scores``) score the LOCAL
    candidates; the gate keeps add-candidates with ``repair_pred >= tau_repair`` and ``sigmoid(edit_logit) >=
    tau_edit`` and remove-candidates with ``safety_pred <= tau_safety`` and ``sigmoid(edit_logit) >= tau_edit``;
    the gated flips are applied by the budget-safe mutual decode. Zero gated flips -> the anchor exactly. No
    evaluator / true-CSI / critic in this path.
    """
    ends = _ends(obs["context"])
    budgets = dict(node_budgets_for_scene(obs["context"].evaluator.scene))
    anchor = _anchor(obs, prev_topology)
    edge_ids = obs["edge_ids"]
    add_ids, remove_ids = local_candidates(anchor, edge_ids, ends, budgets)

    nf_s, ef_s = _standardize(obs["nf"], obs["ef"], mean, std)
    with torch.no_grad():
        out, _h = actor.edit_scores(nf_s, ef_s, obs["ei"])
    idx = {e: i for i, e in enumerate(edge_ids)}
    edit_p = torch.sigmoid(out["edit_logit"])
    repair = out["repair_pred"]
    safety = out["safety_pred"]

    gated_add = [e for e in add_ids
                 if float(edit_p[idx[e]]) >= tau_edit and float(repair[idx[e]]) >= tau_repair]
    gated_remove = [e for e in remove_ids
                    if float(edit_p[idx[e]]) >= tau_edit and float(safety[idx[e]]) <= tau_safety]
    gated_flips = gated_add + gated_remove

    # rank added/removed edges (within budget) by the edit logit -- the head's own confidence
    rank_logits = [float(out["edit_logit"][idx[e]]) for e in edge_ids]
    accept, topology = residual_decode_from_flips(anchor, gated_flips, rank_logits, edge_ids,
                                                  obs["context"], mode=mode)

    n_cand = len(add_ids) + len(remove_ids)
    applied = sorted(set(topology) ^ set(anchor))                 # edges that actually changed vs the anchor
    edit_rate = (len(applied) / n_cand) if n_cand > 0 else 0.0
    return {"anchor": list(anchor), "topology": topology, "accept": accept,
            "gated_add": gated_add, "gated_remove": gated_remove, "gated_flips": gated_flips,
            "applied": applied, "n_candidates": n_cand, "n_add_cand": len(add_ids),
            "n_remove_cand": len(remove_ids), "edit_rate": float(edit_rate),
            "changed": bool(set(topology) != set(anchor))}
