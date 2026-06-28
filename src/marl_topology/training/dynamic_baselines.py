"""D7: fair dynamic baselines -- DEPLOYABLE policies vs CENTRAL references (Contract v3 §10.1/§10.2).

A DEPLOYABLE baseline selects its action from a node's OWN local observation only (incident-edge
features + budget) and the same mutual-acceptance decoder used in deployment -- it NEVER calls the
evaluator / a solver / a searcher / a global decoder at action time (``action_evaluator_calls == 0``).

A CENTRAL reference (e.g. the per-frame myopic-greedy over named candidate topologies) calls the
EVALUATOR to score candidates when selecting its action -- it is an oracle/reference, NOT a deployable
baseline, and must be grouped separately (Contract §10.1: never call a central evaluator-greedy a
"deployable simple baseline").

Both kinds are SCORED with the same dynamic metric (the discounted, hold-scaled episode return +
feasibility/switches, D2/D3) -- that one metric evaluation per frame is identical across methods and is
NOT counted as an action evaluator call. ``baseline_budget_report`` groups the methods and records the
per-method action-evaluator-call budget (Contract §10.2). Gate-exempt ``training/``; evaluation-time
only (no deployed-path change).
"""

from __future__ import annotations

from marl_topology.training.decentralized_action import incident_index
from marl_topology.training.dynamic_rl import _budgets_edges

# graph_payload edge feature 0 is the per-edge link success probability (psucc).
_PSUCC_COL = 0


def _mutual(accept: dict, edge_ids, edges) -> list:
    """An edge is active iff BOTH endpoints proposed it (the deployed mutual-acceptance decoder)."""
    return [edge_ids[i] for i, eid in enumerate(edge_ids)
            if i in accept.get(edges[eid][0], ()) and i in accept.get(edges[eid][1], ())]


def local_threshold_action(ef, edge_ids, edges, budgets, prev_topo, *, threshold, psucc_col=_PSUCC_COL):
    """DEPLOYABLE: each node proposes its incident edges with link-success-prob >= ``threshold`` (its
    top-``budget`` by psucc); an edge is active iff both endpoints propose it. Uses ONLY the local edge
    features ``ef`` -- no evaluator/solver/global decoder."""
    incident = incident_index(edge_ids, edges)
    accept: dict = {}
    for node, idxs in incident.items():
        b = int(budgets.get(node, 0))
        cands = sorted(((float(ef[i, psucc_col]), i) for i in idxs if float(ef[i, psucc_col]) >= threshold),
                       key=lambda t: -t[0])
        accept[node] = {i for _p, i in cands[:b]}
    return _mutual(accept, edge_ids, edges)


def local_hysteresis_proposals(ef, edge_ids, edges, budgets, prev_topo, *, keep_threshold, add_threshold,
                               psucc_col=_PSUCC_COL):
    """The per-node ACCEPT proposals (node -> set of GLOBAL incident edge indices it proposes, |.|<=b_i)
    AND the decoded mutual topology of the deployable hysteresis anchor. The Q5 imitation TARGET: the
    actor learns to PROPOSE these subsets so the local mutual decoder reproduces the anchor. Local
    features + own previous topology only -> 0 evaluator calls."""
    prev = set(prev_topo)
    incident = incident_index(edge_ids, edges)
    accept: dict = {}
    for node, idxs in incident.items():
        b = int(budgets.get(node, 0))
        kept = sorted(((float(ef[i, psucc_col]), i) for i in idxs
                       if edge_ids[i] in prev and float(ef[i, psucc_col]) >= keep_threshold), key=lambda t: -t[0])
        new = sorted(((float(ef[i, psucc_col]), i) for i in idxs
                      if edge_ids[i] not in prev and float(ef[i, psucc_col]) >= add_threshold), key=lambda t: -t[0])
        chosen = (kept[:b] + new)[:b]
        accept[node] = {i for _p, i in chosen}
    return accept, _mutual(accept, edge_ids, edges)


def local_hysteresis_action(ef, edge_ids, edges, budgets, prev_topo, *, keep_threshold, add_threshold,
                            psucc_col=_PSUCC_COL):
    """DEPLOYABLE: keep previous-frame edges still above ``keep_threshold``; fill the remaining budget
    with the best NEW edges above ``add_threshold`` (a keep/repair rule that reduces switching). Uses
    ONLY local features + the node's own previous topology -- no evaluator. (Behavior unchanged; now via
    :func:`local_hysteresis_proposals`.)"""
    return local_hysteresis_proposals(ef, edge_ids, edges, budgets, prev_topo,
                                      keep_threshold=keep_threshold, add_threshold=add_threshold,
                                      psucc_col=psucc_col)[1]


def _score_episode(scene, topo_at, *, reward_of, ref_energy, lam_c, lam_b, beta, reward_mode, eval_for_action):
    """Roll a policy over a scene and score it with the dynamic metric (discounted hold-scaled return +
    feasibility/switches). ``topo_at(t, obs, prev)`` returns the chosen topology; ``eval_for_action`` is
    a counter dict the action selection increments per evaluator call (0 for deployable)."""
    prev: list = []
    e_ref = None
    ep_ret = 0.0
    discount = 1.0
    feas = n = 0
    switch_sum = 0.0
    for t in range(scene.n_frames):
        obs = scene.observation(t, prev)
        if e_ref is None:
            e_ref = ref_energy(obs)
        topo = topo_at(t, obs, prev, e_ref)
        base_r, _gc, _gb, ok = reward_of(obs, list(topo), e_ref, lam_c, lam_b, beta, reward_mode)  # METRIC
        switches = len(frozenset(prev) ^ frozenset(topo)) if t > 0 else 0
        reconfig = (scene.reconfig.e_edge + scene.reconfig.l_edge) * switches
        ep_ret += discount * (scene.hold_interval * base_r - reconfig)
        discount *= scene.gamma
        feas += int(ok); n += 1; switch_sum += switches
        prev = list(topo)
    return ep_ret, feas, n, switch_sum


def evaluate_deployable_baseline(action_fn, scenes, *, reward_of, ref_energy, lam_c, lam_b, beta,
                                 reward_mode, label):
    """Evaluate a DEPLOYABLE baseline. ``action_fn(ef, edge_ids, edges, budgets, prev) -> [edge_id]``
    selects from LOCAL features only -> ``action_evaluator_calls == 0`` by construction. Scored with the
    same dynamic metric as the learned arms."""
    ret_sum = switch_sum = 0.0
    feas = n_frames = 0
    for scene in scenes:
        def topo_at(t, obs, prev, _e_ref):
            budgets, edges = _budgets_edges(obs["context"])
            return action_fn(obs["ef"], obs["edge_ids"], edges, budgets, prev)
        ep_ret, f, nfr, sw = _score_episode(scene, topo_at, reward_of=reward_of, ref_energy=ref_energy,
                                            lam_c=lam_c, lam_b=lam_b, beta=beta, reward_mode=reward_mode,
                                            eval_for_action=None)
        ret_sum += ep_ret; feas += f; n_frames += nfr; switch_sum += sw
    return {"label": label, "group": "deployable_policy", "action_evaluator_calls": 0,
            "per_frame_feasibility": feas / max(1, n_frames),
            "mean_episode_return": ret_sum / max(1, len(scenes)),
            "mean_switches_per_frame": switch_sum / max(1, n_frames), "n_scenes": len(scenes)}


def evaluate_central_reference(scenes, *, reward_of, ref_energy, lam_c, lam_b, beta, reward_mode,
                               label="myopic_greedy"):
    """Evaluate a CENTRAL reference: per-frame reward-greedy over the named candidate variants. The
    ACTION SELECTION calls the evaluator (via ``reward_of``) to score every candidate -> counted as
    ``action_evaluator_calls`` (> 0). NOT a deployable baseline (Contract §10.1)."""
    calls = {"n": 0}
    ret_sum = switch_sum = 0.0
    feas = n_frames = 0
    for scene in scenes:
        def topo_at(t, obs, prev, e_ref):
            ctx = obs["context"]
            tv = getattr(ctx, "topology_variants", None) or {}
            values = tv.values() if isinstance(tv, dict) else tv
            cands = [tuple(str(e) for e in (v.edges if hasattr(v, "edges") else v)) for v in values]
            cands = cands or [tuple(obs["edge_ids"])]
            best, best_r = cands[0], -1e30
            for cand in cands:
                r, _gc, _gb, _ok = reward_of(obs, list(cand), e_ref, lam_c, lam_b, beta, reward_mode)
                calls["n"] += 1                         # the action selection USES the evaluator
                if r > best_r:
                    best, best_r = cand, r
            return list(best)
        ep_ret, f, nfr, sw = _score_episode(scene, topo_at, reward_of=reward_of, ref_energy=ref_energy,
                                            lam_c=lam_c, lam_b=lam_b, beta=beta, reward_mode=reward_mode,
                                            eval_for_action=calls)
        ret_sum += ep_ret; feas += f; n_frames += nfr; switch_sum += sw
    return {"label": label, "group": "central_reference", "action_evaluator_calls": calls["n"],
            "per_frame_feasibility": feas / max(1, n_frames),
            "mean_episode_return": ret_sum / max(1, len(scenes)),
            "mean_switches_per_frame": switch_sum / max(1, n_frames), "n_scenes": len(scenes)}


def baseline_budget_report(results) -> dict:
    """Group baseline results into deployable_policy / central_reference (and any other group) with the
    per-method action-evaluator-call budget (Contract §10.1/§10.2). Deployable methods MUST be
    evaluator-free at action time."""
    groups: dict = {}
    for r in results:
        groups.setdefault(r["group"], []).append({
            "label": r["label"], "action_evaluator_calls": r["action_evaluator_calls"],
            "per_frame_feasibility": round(r["per_frame_feasibility"], 5),
            "mean_episode_return": round(r["mean_episode_return"], 5),
            "mean_switches_per_frame": round(r.get("mean_switches_per_frame", 0.0), 5),
        })
    return {
        "groups": groups,
        "deployable_use_no_evaluator_for_action": all(
            m["action_evaluator_calls"] == 0 for m in groups.get("deployable_policy", [])),
        "note": ("deployable_policy = local-only action (0 evaluator calls); central_reference = "
                 "evaluator-greedy candidate search (oracle/reference, NOT a deployable baseline)."),
    }
