"""R5 (Belief-Guided Residual PPO): supervised repair/safety/edit heads -- the DEPLOYABLE-LEARNING crux.

R4 proved a beneficial-edit signal EXISTS (true-evaluator-scored). R5 trains heads on ONLY LOCAL features (the
deployed actor's graph encoding of nf/ef -- no evaluator, no true CSI) to predict those edits, and asks
whether the held top-k edit-hit-rate beats a same-budget random baseline. If not, local observations cannot
predict the central beneficial edits at N<=16 -> a path-specific deployable LEARNING gap (the campaign's
honest final diagnosis). The R4 evaluator scores are TRAINING LABELS only; the head input never sees them.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from marl_topology.budgets import node_budgets_for_scene
from marl_topology.models.belief_residual_actor import BeliefResidualActor
from marl_topology.training.dynamic_baselines import local_hysteresis_action
from marl_topology.training.dynamic_rl import _budgets_edges, _standardize
from marl_topology.training.quorum_deficit_bridge import topology_quorum_deficit, topology_reliability
from marl_topology.training.residual_repair import _degrees, _ends
from marl_topology.training.residual_saturation import feature_standardization_all_frames

_KT, _AT = 0.4, 0.6
_LAM_C, _LAM_B, _BETA = 1.0, 1.0, 0.1
_DEFICIT_KEY = "d_quorum_mean"


def _anchor(obs, prev):
    budgets, edges = _budgets_edges(obs["context"])
    return local_hysteresis_action(obs["ef"], obs["edge_ids"], edges, budgets, prev,
                                   keep_threshold=_KT, add_threshold=_AT)


def _d(ev, topo):
    d = topology_quorum_deficit(ev, topo)
    return d[_DEFICIT_KEY] if d else float("inf")


def build_edit_examples(scenes, T, mean, std, *, margin=0.0, tau=0.9):
    """Per frame: the LOCAL obs features (nf_s, ef_s, ei) + per-candidate-edge targets from the R4 evaluator
    scoring (TRAINING LABEL). add-candidates are budget-feasible non-anchor edges; remove-candidates are
    anchor edges. repair_gain = ΔD-reduction for adds; deletion_risk = ΔD-increase for removes."""
    examples = []
    for sc in scenes:
        prev, e_ref = [], None
        for t in range(sc.n_frames):
            obs = sc.observation(t, prev)
            ev = obs["context"].evaluator
            if e_ref is None:
                e_ref = T._ref_energy(obs)
            ends = _ends(obs["context"])
            budgets = dict(node_budgets_for_scene(ev.scene))
            anchor = _anchor(obs, prev)
            anchor_set = set(anchor)
            deg = _degrees(anchor, ends)
            ra = topology_reliability(ev, anchor)
            da = _d(ev, anchor)
            ja = float(T.reward_of(obs, list(anchor), e_ref, _LAM_C, _LAM_B, _BETA, "dense")[0])
            eids = obs["edge_ids"]
            E = len(eids)
            positive = torch.zeros(E); repair = torch.zeros(E); risk = torch.zeros(E)
            cand = torch.zeros(E); add_m = torch.zeros(E); rem_m = torch.zeros(E)
            for i, e in enumerate(eids):
                if e in anchor_set:                                  # REMOVE candidate
                    edited = [x for x in anchor if x != e]; kind = "remove"
                else:                                                # ADD candidate (budget-feasible only)
                    u, v = ends[e]
                    if deg.get(u, 0) >= budgets.get(u, 0) or deg.get(v, 0) >= budgets.get(v, 0):
                        continue
                    edited = list(anchor) + [e]; kind = "add"
                re = topology_reliability(ev, edited)
                de = _d(ev, edited)
                je = float(T.reward_of(obs, list(edited), e_ref, _LAM_C, _LAM_B, _BETA, "dense")[0])
                dj = je - ja
                safe = not (ra["consensus"] >= tau and re["consensus"] < tau)
                cand[i] = 1.0
                positive[i] = 1.0 if (dj > margin and safe) else 0.0
                if kind == "add":
                    add_m[i] = 1.0
                    repair[i] = float(da - de) if (da != float("inf") and de != float("inf")) else 0.0
                else:
                    rem_m[i] = 1.0
                    risk[i] = float(de - da) if (da != float("inf") and de != float("inf")) else 0.0
            nf_s, ef_s = _standardize(obs["nf"], obs["ef"], mean, std)
            examples.append({"nf_s": nf_s, "ef_s": ef_s, "ei": obs["ei"], "positive": positive,
                             "repair": repair, "risk": risk, "cand": cand, "add_m": add_m, "rem_m": rem_m})
            prev = list(anchor)
    return examples


def _masked_bce(logit, target, mask):
    if mask.sum() == 0:
        return logit.new_zeros(())
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    return (bce * mask).sum() / mask.sum()


def _masked_huber(pred, target, mask):
    if mask.sum() == 0:
        return pred.new_zeros(())
    h = F.smooth_l1_loss(pred, target, reduction="none")
    return (h * mask).sum() / mask.sum()


def topk_metrics(examples, actor):
    """Held top-k beneficial-edit precision (k = #positive per frame) vs the same-budget RANDOM base rate,
    + repair/safety prediction correlation. Higher precision than base rate => the heads learned a deployable
    direction signal."""
    prec, base, lifts, rep_corr, saf_corr = [], [], [], [], []
    for ex in examples:
        with torch.no_grad():
            out, _h = actor.edit_scores(ex["nf_s"], ex["ef_s"], ex["ei"])
        cand = ex["cand"] > 0
        n_cand = int(cand.sum())
        pos = ex["positive"][cand]
        n_pos = int(pos.sum())
        if n_cand == 0 or n_pos == 0:
            continue
        scores = out["edit_logit"][cand]
        k = n_pos
        topk = torch.topk(scores, min(k, n_cand)).indices
        prec.append(float(pos[topk].sum()) / k)                      # precision@k (k = #positive)
        base.append(n_pos / n_cand)                                  # random base rate
        lifts.append(prec[-1] / max(1e-9, base[-1]))
        addm = ex["add_m"][cand] > 0
        remm = ex["rem_m"][cand] > 0
        if int(addm.sum()) >= 2:
            rep_corr.append(_corr(out["repair_pred"][cand][addm], ex["repair"][cand][addm]))
        if int(remm.sum()) >= 2:
            saf_corr.append(_corr(out["safety_pred"][cand][remm], ex["risk"][cand][remm]))
    def _m(x):
        x = [v for v in x if v == v]
        return round(sum(x) / len(x), 5) if x else None
    return {"topk_precision": _m(prec), "random_base_rate": _m(base), "lift": _m(lifts),
            "precision_minus_base": _m([p - b for p, b in zip(prec, base)]),
            "repair_corr": _m(rep_corr), "safety_corr": _m(saf_corr), "n_frames_scored": len(prec)}


def _corr(a, b):
    a = a.detach().float(); b = b.detach().float()
    if a.numel() < 2 or float(a.std()) < 1e-9 or float(b.std()) < 1e-9:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std() + 1e-9))


def train_edit_heads(train_scenes, held_scenes, T, *, epochs=40, hidden=64, lr=0.02, margin=0.0, seed=0):
    """Train the edit/repair/safety heads on LOCAL features (R4 targets as labels); eval on held."""
    torch.manual_seed(seed)
    mean, std = feature_standardization_all_frames(train_scenes)
    nd = train_scenes[0].observation(0, [])["nf"].shape[1]
    ed = train_scenes[0].observation(0, [])["ef"].shape[1]
    actor = BeliefResidualActor(nd, ed, hidden=hidden)
    params = list(actor.edit_head.parameters()) + list(actor.repair_head.parameters()) + \
        list(actor.safety_head.parameters()) + list(actor.node_enc.parameters()) + \
        list(actor.msg.parameters()) + list(actor.gru.parameters()) + list(actor.h_norm.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    train_ex = build_edit_examples(train_scenes, T, mean, std, margin=margin)
    held_ex = build_edit_examples(held_scenes, T, mean, std, margin=margin)
    untrained = topk_metrics(held_ex, actor)
    for _e in range(epochs):
        loss = torch.zeros(())
        for ex in train_ex:
            out, _h = actor.edit_scores(ex["nf_s"], ex["ef_s"], ex["ei"])
            loss = loss + _masked_bce(out["edit_logit"], ex["positive"], ex["cand"]) \
                + _masked_huber(out["repair_pred"], ex["repair"], ex["add_m"]) \
                + _masked_huber(out["safety_pred"], ex["risk"], ex["rem_m"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    trained = topk_metrics(held_ex, actor)
    return {"untrained_held": untrained, "trained_held": trained, "n_train_examples": len(train_ex),
            "n_held_examples": len(held_ex), "actor": actor, "mean": mean, "std": std, "hidden": hidden}
