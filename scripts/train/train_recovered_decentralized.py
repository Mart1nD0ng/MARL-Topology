"""Retrain the recovered decentralized trunk (critic-planner -> BC-distilled actor).

The full validated recipe, recovered into src (`docs/URBAN_V2X_RESEARCH_LOG.md` phase2p / Step-3):

  load operating-point dataset shards -> 60/40 split -> fit a centralized graph critic and harden
  it with DAgger dense hard-negatives (select by held-out controller feasibility) -> generate
  evaluator-verified critic-PLANNER targets for the train scenes -> BC-distil the K-round
  message-passing actor on those targets (weight decay + early stopping + keep-best) -> evaluate
  held-out with the DECENTRALIZED local mutual-acceptance decoder.

Artifacts (frozen actors + critic + norm + result JSON) are written under result_save/. Defaults
run a modest CPU retrain; pass larger seeds/sizes for the full 3-seed campaign or on a GPU. The
training body lives in ``run_campaign_arm`` so the 4090 campaign orchestrator
(``scripts/train/run_4090_campaign.py``) can reuse it per regime without re-loading shards.

Usage::

    python scripts/train/train_recovered_decentralized.py \
        --shards result_save/recovered_step3/_step3_shard_3001.pkl ... 3004.pkl \
        --out-dir result_save/decentralized_retrain
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from random import Random
from statistics import fmean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from marl_topology.training.critic_guided_planner import (  # noqa: E402
    build_critic_samples,
    collect_dense_hard_negatives,
    controller_feasibility,
    fit_critic,
    planner_targets,
    standardization_state,
)
from marl_topology.training.decentralized_distillation import (  # noqa: E402
    build_samples,
    ci95,
    feasible_breakdown,
    feasible_rate_by_n,
    global_argsort_assemble,
    local_mutual_assemble,
    train_actor,
)
from marl_topology.training.production_mappo_adapter import (  # noqa: E402
    Stage33ProductionMappoAdapter,
)

DEFAULT_SHARDS = [str(ROOT / "result_save" / "recovered_step3" / f"_step3_shard_{s}.pkl")
                  for s in (3001, 3002, 3003, 3004)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards", nargs="+", default=DEFAULT_SHARDS)
    p.add_argument("--out-dir", default=str(ROOT / "result_save" / "decentralized_retrain"))
    p.add_argument("--held-frac", type=float, default=0.4)
    p.add_argument("--split-seed", type=int, default=7)
    p.add_argument("--critic-seeds", type=int, nargs="+", default=[0])
    p.add_argument("--dagger-iters", type=int, default=1)
    p.add_argument("--dagger-scenes", type=int, default=40)
    p.add_argument("--val-scenes", type=int, default=12)
    p.add_argument("--critic-epochs", type=int, default=40)
    p.add_argument("--beam-width", type=int, default=6)
    p.add_argument("--verify-k", type=int, default=20)
    p.add_argument("--actor-seeds", type=int, nargs="+", default=[0])
    p.add_argument("--train-size", type=int, default=0, help="0 = all available train scenes")
    # Actor capacity. Exposed for the K-rounds architecture ablation (rounds=0 collapses the
    # K-round message-passing actor to a per-edge MLP -> "is it coordinated message passing or a
    # bandit?"). Defaults reproduce the validated Step-3 actor.
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--hidden", type=int, default=64)
    return p.parse_args()


def load_pool(shard_paths):
    adapter = Stage33ProductionMappoAdapter()
    pool = []
    for shard_path in shard_paths:
        with open(shard_path, "rb") as handle:
            dataset = pickle.load(handle)
        labels = dataset.source_dataset.teacher_labels
        for split in ("train", "eval", "test"):
            for row, context in adapter.build_row_contexts(dataset, split):
                pool.append((row, context, labels[context.fixture.fixture_id]))
    return pool


def run_campaign_arm(
    pool,
    *,
    held_frac=0.4,
    split_seed=7,
    critic_seeds=(0,),
    dagger_iters=1,
    dagger_scenes=40,
    val_scenes=12,
    critic_epochs=40,
    beam_width=6,
    verify_k=20,
    actor_seeds=(0,),
    train_size=0,
    rounds=4,
    hidden=64,
    use_planner=True,
    log=print,
):
    """Run one teacher -> BC-distilled-actor arm on a pre-loaded pool.

    With ``use_planner=True`` (default) the validated recipe: fit + DAgger-harden a centralized
    critic, generate evaluator-verified critic-PLANNER targets, BC-distil the actor on them. With
    ``use_planner=False`` the actor is BC-distilled directly on the raw SA-teacher labels (no
    critic) -- the E8 ablation isolating the planner-target lift over the SA teacher.

    Returns a dict with a JSON-serializable ``results`` block (raw + solvable-conditional held-out
    feasibility, per-N, decentralization cost) and the frozen ``actor_states`` / ``critic_state`` /
    ``norm_state`` for persisting. Pure compute -- the caller owns I/O.
    """

    pool = list(pool)
    Random(split_seed).shuffle(pool)
    cut = int((1.0 - held_frac) * len(pool))
    train_items, held_items = pool[:cut], pool[cut:]
    val_items = train_items[-val_scenes:]
    collect_items = train_items[:-val_scenes]
    ceiling = fmean(float(bool(l["feasible_exists"])) for _r, _c, l in held_items)
    log(f"[data] pool {len(pool)} | train {len(train_items)} | held {len(held_items)} | "
        f"teacher ceiling (held) {ceiling:.3f}")

    critic_state = critic_tag = norm_state = None
    if use_planner:
        # critic candidates, hardened with DAgger, selected by held-out (val) controller feasibility
        best = {"score": -1.0, "state": None, "norm": None, "tag": None}
        for cseed in critic_seeds:
            srng = Random(100 + cseed)
            base = build_critic_samples([(r, c) for r, c, _l in train_items], srng)
            norm = standardization_state([s[0] for s in base])
            critic = fit_critic(base, norm, cseed, epochs=critic_epochs)
            samples = list(base)
            for it in range(1, dagger_iters + 1):
                subset = srng.sample(collect_items, min(dagger_scenes, len(collect_items)))
                samples += collect_dense_hard_negatives(critic, norm, subset, beam_width=beam_width)
                critic = fit_critic(samples, norm, cseed, epochs=critic_epochs)
                _pure, verify = controller_feasibility(critic, norm, val_items,
                                                       verify_k=verify_k, beam_width=beam_width)
                log(f"[critic] seed {cseed} iter {it}: VAL controller feasibility {verify:.3f}")
                if verify > best["score"]:
                    best = {"score": verify,
                            "state": {k: v.detach().clone() for k, v in critic.state_dict().items()},
                            "norm": norm, "tag": f"s{cseed}i{it}"}
        log(f"[critic] SELECTED {best['tag']}: val controller feasibility {best['score']:.3f}")

        from marl_topology.models import CentralizedMessagePassingGraphCritic
        from marl_topology.models.centralized_message_passing_graph_critic import (
            CentralizedMessagePassingGraphCriticConfig,
        )
        critic = CentralizedMessagePassingGraphCritic(CentralizedMessagePassingGraphCriticConfig())
        critic.load_state_dict(best["state"]); critic.eval()

        # critic-planner targets for the train scenes
        retargeted, feas = planner_targets(critic, best["norm"], train_items,
                                           verify_k=verify_k, beam_width=beam_width)
        log(f"[planner] feasible train targets: {feas}/{len(train_items)} "
            f"({feas / max(1, len(train_items)):.3f})")
        critic_state, critic_tag, norm_state = best["state"], best["tag"], best["norm"]
    else:
        # E8 ablation: BC-distil directly on the raw SA-teacher labels (no critic / no planner)
        retargeted = list(train_items)
        feas = sum(int(bool(l["feasible_exists"])) for _r, _c, l in train_items)
        log(f"[teacher] SA-teacher targets (no planner): feasible {feas}/{len(train_items)} "
            f"({feas / max(1, len(train_items)):.3f})")

    # --- BC-distil the decentralized actor on the (planner or SA-teacher) targets
    held = build_samples(held_items)
    train_size = train_size or (len(retargeted) - 8)
    results = {"ceiling": ceiling, "critic_tag": critic_tag, "rounds": rounds, "hidden": hidden,
               "use_planner": use_planner,
               "planner_train_feasible": feas / max(1, len(train_items)),
               "dec": [], "glob": [], "by_n": [], "breakdown": []}
    actor_states = []
    for seed in actor_seeds:
        sub = Random(1000 + seed).sample(retargeted, min(train_size, len(retargeted)))
        n_val = min(max(4, int(0.15 * len(sub))), max(1, len(sub) - 1))
        actor, mean, std, n_ep, _vb = train_actor(
            build_samples(sub[n_val:]), build_samples(sub[:n_val]), seed, hidden=hidden, rounds=rounds)
        dec, dec_by_n = feasible_rate_by_n(actor, held, mean, std, local_mutual_assemble)
        glo, _ = feasible_rate_by_n(actor, held, mean, std, global_argsort_assemble)
        bd = feasible_breakdown(actor, held, mean, std, local_mutual_assemble)
        results["dec"].append(dec); results["glob"].append(glo); results["by_n"].append(dec_by_n)
        results["breakdown"].append(bd)
        actor_states.append({"state": actor.state_dict(), "mean": mean, "std": std})
        cond = bd["conditional"]
        log(f"[actor] seed {seed}: held DEC={dec:.3f} (by N {dec_by_n}) "
            f"conditional={cond if cond is None else round(cond, 3)} global={glo:.3f} (ep={n_ep})")

    dm, dh = ci95(results["dec"])
    gm, gh = ci95(results["glob"])
    cond_vals = [bd["conditional"] for bd in results["breakdown"] if bd["conditional"] is not None]
    cm, ch = ci95(cond_vals) if cond_vals else (None, 0.0)
    results.update({"dec_mean": dm, "dec_ci95": dh, "glob_mean": gm, "glob_ci95": gh,
                    "decentralization_cost": gm - dm,
                    "conditional_mean": cm, "conditional_ci95": ch})
    log("=" * 72)
    log(f"[arm] held-out DECENTRALIZED raw = {dm:.3f} +/- {dh:.3f}  (teacher ceiling {ceiling:.3f})")
    if cm is not None:
        log(f"[arm] held-out DECENTRALIZED conditional (solved/solvable) = {cm:.3f} +/- {ch:.3f}")
    log(f"[arm] held-out global ablation = {gm:.3f} +/- {gh:.3f}  (decentralization cost {gm - dm:+.3f})")
    return {"results": results, "actor_states": actor_states,
            "critic_state": critic_state, "critic_tag": critic_tag, "norm_state": norm_state}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = load_pool(args.shards)
    arm = run_campaign_arm(
        pool,
        held_frac=args.held_frac, split_seed=args.split_seed,
        critic_seeds=args.critic_seeds, dagger_iters=args.dagger_iters,
        dagger_scenes=args.dagger_scenes, val_scenes=args.val_scenes,
        critic_epochs=args.critic_epochs, beam_width=args.beam_width, verify_k=args.verify_k,
        actor_seeds=args.actor_seeds, train_size=args.train_size,
        rounds=args.rounds, hidden=args.hidden,
    )
    torch.save({"critic_state": arm["critic_state"], "critic_tag": arm["critic_tag"],
                "norm_state": arm["norm_state"], "actors": arm["actor_states"]},
               out_dir / "_artifacts_retrain.pt")
    (out_dir / "retrain_result.json").write_text(
        json.dumps(arm["results"], indent=2, default=str), encoding="utf-8")
    print(f"[done] artifacts + result in {out_dir}")


if __name__ == "__main__":
    main()
