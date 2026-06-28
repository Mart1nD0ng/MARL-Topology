"""Q5 (POMDP-QP-FAR): can the actor REPRODUCE the deployable local_hysteresis anchor? (Spec S13 Stage B)

Warm-starts a DynamicRecurrentActor (memoryless) toward the DEPLOYABLE local_hysteresis anchor's per-node
BCSP proposals (0 evaluator calls -- the anchor, NOT the central myopic teacher), then measures on HELD
scenes: BCSP-subset NLL (does BC converge?), decoded-topology F1 vs the anchor (teacher-forced, same
state), held feasibility (actor's deployed rollout vs the anchor's), and switches/frame. If the actor
cannot reproduce the anchor, residual RL (Q6) must NOT start. Eval-only: no reward, no PBRS, no residual.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

import torch  # noqa: E402

from marl_topology.models.dynamic_recurrent_actor import DynamicRecurrentActor  # noqa: E402
from marl_topology.policies.decentralized_mutual_acceptance import local_mutual_assemble  # noqa: E402
from marl_topology.training.decentralized_distillation import feature_standardization  # noqa: E402
from marl_topology.training.dynamic_baselines import local_hysteresis_proposals  # noqa: E402
from marl_topology.training.dynamic_rl import (  # noqa: E402
    _budgets_edges, _hysteresis_teacher_trajectory, _standardize, _teacher_subset_nll)
from marl_topology.training.two_timescale_env import ReconfigCost  # noqa: E402


def _load_trunk():
    spec = importlib.util.spec_from_file_location(
        "trunk", ROOT / "scripts" / "train" / "train_decentralized_rl.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def topology_f1(a, b) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    prec, rec = inter / len(a), inter / len(b)
    return 2 * prec * rec / (prec + rec)


def warmstart(actor, teachers, mean, std, *, epochs, lr, temp):
    def mean_nll():
        with torch.no_grad():
            vals = [float(_teacher_subset_nll(actor, t, mean, std, recurrent=False, temp=temp)) for t in teachers]
        return sum(vals) / max(1, len(vals))
    init = mean_nll()
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    for _ep in range(max(1, epochs)):
        for traj in teachers:
            loss = _teacher_subset_nll(actor, traj, mean, std, recurrent=False, temp=temp)
            if not loss.requires_grad:
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
    return init, mean_nll()


def eval_imitation(actor, scenes, mean, std, *, T, keep_threshold, add_threshold, temp):
    lam_c = lam_b = 1.0
    beta = 0.1
    f1s = []
    actor_feas = anchor_feas = nfr = 0
    actor_sw = anchor_sw = 0.0
    for sc in scenes:
        a_prev: list = []
        h_prev: list = []
        e_ref = None
        for t in range(sc.n_frames):
            obs_h = sc.observation(t, h_prev)                       # the anchor's own rollout
            budgets, edges = _budgets_edges(obs_h["context"])
            if e_ref is None:
                e_ref = T._ref_energy(obs_h)
            _acc, anchor_topo = local_hysteresis_proposals(
                obs_h["ef"], obs_h["edge_ids"], edges, budgets, h_prev,
                keep_threshold=keep_threshold, add_threshold=add_threshold)
            _r, _gc, _gb, h_ok = T.reward_of(obs_h, list(anchor_topo), e_ref, lam_c, lam_b, beta, "dense")
            # teacher-forced F1: the actor on the SAME state the anchor saw
            nf_s, ef_s = _standardize(obs_h["nf"], obs_h["ef"], mean, std)
            with torch.no_grad():
                logits, _ = actor(nf_s, ef_s, obs_h["ei"], hidden=None)
            tf_topo = local_mutual_assemble(logits, obs_h["edge_ids"], obs_h["context"])
            f1s.append(topology_f1(tf_topo, anchor_topo))
            # deployed actor rollout (own previous topology)
            obs_a = sc.observation(t, a_prev)
            nf_a, ef_a = _standardize(obs_a["nf"], obs_a["ef"], mean, std)
            with torch.no_grad():
                logits_a, _ = actor(nf_a, ef_a, obs_a["ei"], hidden=None)
            actor_topo = local_mutual_assemble(logits_a, obs_a["edge_ids"], obs_a["context"])
            _r2, _g2, _g3, a_ok = T.reward_of(obs_a, list(actor_topo), e_ref, lam_c, lam_b, beta, "dense")
            anchor_feas += int(h_ok); actor_feas += int(a_ok); nfr += 1
            if t > 0:
                anchor_sw += len(frozenset(h_prev) ^ frozenset(anchor_topo))
                actor_sw += len(frozenset(a_prev) ^ frozenset(actor_topo))
            h_prev = list(anchor_topo); a_prev = list(actor_topo)
    return {"decoded_f1_teacher_forced": round(sum(f1s) / max(1, len(f1s)), 4),
            "actor_held_feasibility": round(actor_feas / max(1, nfr), 4),
            "anchor_held_feasibility": round(anchor_feas / max(1, nfr), 4),
            "actor_switches_per_frame": round(actor_sw / max(1, nfr), 4),
            "anchor_switches_per_frame": round(anchor_sw / max(1, nfr), 4), "n_frames": nfr}


def _build(data, seed, count, args):
    from build_operating_point_dataset import operating_point_regime
    from marl_topology.training.dynamic_frames import (
        sample_dynamic_scenes, sample_dynamic_urban_scenes)
    common = dict(seed=seed, count=count, node_count_choices=tuple(args.dyn_nodes),
                  regime=operating_point_regime(args.tx_power), num_frames=args.frames, dt_s=args.dt,
                  speed_min_mps=args.speed_min, speed_max_mps=args.speed_max, reconfig=ReconfigCost(),
                  hold_interval=args.hold_interval, gamma=args.gamma)
    if data == "urban":
        return sample_dynamic_urban_scenes(rsu_count=4, blocks_per_side=3, **common)
    return sample_dynamic_scenes(**common)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["random", "urban"])
    ap.add_argument("--train", type=int, default=16)
    ap.add_argument("--held", type=int, default=10)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--dt", type=float, default=2.0)
    ap.add_argument("--speed-min", type=float, default=15.0)
    ap.add_argument("--speed-max", type=float, default=30.0)
    ap.add_argument("--hold-interval", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--dyn-nodes", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--tx-power", type=float, default=20.0)
    ap.add_argument("--keep-threshold", type=float, default=0.4)
    ap.add_argument("--add-threshold", type=float, default=0.6)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=4505)
    ap.add_argument("--out", default=str(ROOT / "result_save" / "anchor_imitation.json"))
    args = ap.parse_args()

    T = _load_trunk()
    torch.manual_seed(args.seed)
    report = {"scope": "Q5 local_hysteresis imitation (eval-only); anchor=deployable 0-eval-call",
              "config": {"train": args.train, "held": args.held, "frames": args.frames,
                         "epochs": args.epochs, "keep_threshold": args.keep_threshold,
                         "add_threshold": args.add_threshold, "seed": args.seed}, "by_data": {}}
    for data in args.data:
        train = _build(data, args.seed * 1000 + 1, args.train, args)
        held = _build(data, args.seed * 1000 + 777, args.held, args)
        teachers = [_hysteresis_teacher_trajectory(s, keep_threshold=args.keep_threshold,
                                                   add_threshold=args.add_threshold) for s in train]
        stat = [s.observation(0, []) for s in train]
        mean, std = feature_standardization(stat)
        node_dim, edge_dim = stat[0]["nf"].shape[1], stat[0]["ef"].shape[1]
        actor = DynamicRecurrentActor(node_dim, edge_dim, hidden=args.hidden)
        init_nll, final_nll = warmstart(actor, teachers, mean, std, epochs=args.epochs, lr=args.lr, temp=args.temp)
        ev = eval_imitation(actor, held, mean, std, T=T, keep_threshold=args.keep_threshold,
                            add_threshold=args.add_threshold, temp=args.temp)
        row = {"teacher_nll_init": round(init_nll, 4), "teacher_nll_final": round(final_nll, 4), **ev}
        report["by_data"][data] = row
        print(f"[{data}] NLL {init_nll:.3f}->{final_nll:.3f}  F1={ev['decoded_f1_teacher_forced']} "
              f"actor_feas={ev['actor_held_feasibility']} anchor_feas={ev['anchor_held_feasibility']} "
              f"actor_sw={ev['actor_switches_per_frame']} anchor_sw={ev['anchor_switches_per_frame']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
