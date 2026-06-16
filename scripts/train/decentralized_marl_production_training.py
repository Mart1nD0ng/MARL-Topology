"""Production training driver for the decentralized CTDE MARL trunk.

Runs the decentralized multi-agent policy-gradient flow (K-hop local actor + per-node
mutual-acceptance decoder + training-only centralized critic) over procedurally generated
moving-vehicle episodes, optionally sweeping the receptive-field depth K as the headline
architecture ablation. Saves per-config checkpoints, a training report, result figures, and
data tables.

This is the version-controlled production training driver. Real large-scale training +
checkpointing is authorized (post-2026-06-16), so this driver legitimately imports torch and
writes checkpoints (the scaffold-era "no training / report-only" assumption is retired). It is
scale-parameterized: the defaults run a small CPU smoke locally; pass ``--device cuda`` and
larger ``--scenario-count/--updates/--node-choices`` on a rented GPU (e.g. a 4090) for the
large-scale run.

Examples
--------
Local smoke (CPU, fast)::

    python scripts/train/decentralized_marl_production_training.py --smoke

Large-scale (rented 4090)::

    python scripts/train/decentralized_marl_production_training.py \
        --device cuda --rounds 1 2 3 --scenario-count 4000 --updates 800 \
        --node-choices 4 6 8 10 12 16 --large-n 24 32 --rollout-steps 24 \
        --out-dir result_save/decentralized_marl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Windows torch+matplotlib both link an OpenMP runtime; allow the duplicate so figure
# generation does not abort with "OMP Error #15". Safe for this offline plotting driver.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Some containers (e.g. autodl) ship an INVALID OMP_NUM_THREADS, which makes libgomp abort
# ("Invalid value for environment variable OMP_NUM_THREADS") and can wedge OpenMP+fork
# workers. Force a sane value before torch/numpy initialise OpenMP. 1 thread per process is
# also correct here: parallelism is across rollout worker PROCESSES, not OpenMP threads.
_omp = os.environ.get("OMP_NUM_THREADS", "")
if not _omp.isdigit() or int(_omp) < 1:
    os.environ["OMP_NUM_THREADS"] = "1"

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from marl_topology.data.stage31_scenario_generator import (  # noqa: E402
    ProductionScenarioConfig,
    generate_production_scenarios,
)
from marl_topology.training.decentralized_marl import (  # noqa: E402
    DecentralizedCTDEFlow,
    DecentralizedMARLConfig,
    decentralized_production_gate,
)
from marl_topology.training.production_trunk import production_trunk_designation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--seed", type=int, default=4096)
    parser.add_argument("--scenario-count", type=int, default=400)
    parser.add_argument("--eval-frac", type=float, default=0.2)
    parser.add_argument("--node-choices", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    parser.add_argument("--large-n", type=int, nargs="*", default=[24, 32],
                        help="held-out larger node counts for the scale-generalization test")
    parser.add_argument("--large-n-count", type=int, default=60)
    parser.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3],
                        help="receptive-field depths K to ablate (K=1 ~ ego-graph baseline)")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--scenes-per-update", type=int, default=128,
                        help="parallel-env batch rolled per update (0 = whole train pool)")
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--warmup-updates", type=int, default=15,
                        help="behaviour-cloning warm-start passes distilling the SA teacher")
    parser.add_argument("--bc-sa-iters", type=int, default=120,
                        help="SA iterations for the BC teacher search (per restart)")
    parser.add_argument("--bc-restarts", type=int, default=4, help="SA restarts for the BC teacher")
    parser.add_argument("--energy-budget", type=float, default=30.0,
                        help="per-node battery CAPACITY (J), full at start")
    parser.add_argument("--energy-recharge", type=float, default=-1.0,
                        help="per-step energy recharge (J); -1 = auto-size to the max hub-star degree "
                             "so the feasible RSU star is sustainable across the whole episode")
    parser.add_argument("--entropy-coef", type=float, default=0.0,
                        help="PPO entropy bonus (default 0: a bonus erodes the peaked BC warm start)")
    parser.add_argument("--churn-weight", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="parallel rollout workers (0 = serial; use >0 on a Linux box)")
    parser.add_argument("--no-vectorized-evaluator", action="store_true",
                        help="use the canonical (slower) Stage-21 evaluator instead of the vectorized one")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--out-dir", default="result_save/decentralized_marl")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny fast run for local correctness validation")
    return parser.parse_args()


def _smoke(args: argparse.Namespace) -> argparse.Namespace:
    args.scenario_count = 10
    args.node_choices = [4, 5]
    args.large_n = [8]
    args.large_n_count = 4
    args.rounds = [1, 3]
    args.hidden_dim = 32
    args.updates = 2
    args.warmup_updates = 2
    args.rollout_steps = 3
    args.minibatch_size = 8
    args.update_epochs = 1
    args.eval_every = 1
    args.bc_sa_iters = 20
    args.bc_restarts = 2
    return args


def _generate(node_choices, count, seed):
    return list(
        generate_production_scenarios(
            ProductionScenarioConfig(
                seed=seed, scenario_count=count, node_count_choices=tuple(node_choices)
            )
        )
    )


def main() -> None:
    args = parse_args()
    if args.smoke:
        args = _smoke(args)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = _generate(args.node_choices, args.scenario_count, args.seed)
    eval_count = max(1, int(len(specs) * args.eval_frac))
    eval_specs = specs[:eval_count]
    train_specs = specs[eval_count:]
    large_n_specs = _generate(args.large_n, args.large_n_count, args.seed + 99) if args.large_n else []
    print(f"[data] {len(train_specs)} train / {len(eval_specs)} eval / {len(large_n_specs)} held-out "
          f"large-N scenes; N in {sorted(args.node_choices)}; large-N {sorted(args.large_n)}; device={args.device}")

    # Recharging power-budget sizing: the unique feasible PBFT structure is a degree-(N-1) RSU
    # star, so the hub must afford degree (N-1) EVERY step. Recharge >= max-hub-degree * unit makes
    # that sustainable across the whole episode (the one-shot battery drained the hub in ~2 steps).
    # Capacity = 2 * recharge gives a one-step burst buffer whose drain still couples to the next
    # step (keeps it genuinely non-bandit). Auto-sized from the trained + held-out node counts.
    energy_unit_j = 1.0
    all_n = list(args.node_choices) + list(args.large_n or [])
    max_hub_degree = (max(all_n) - 1) if all_n else 1
    recharge_j = args.energy_recharge if args.energy_recharge >= 0.0 else float(max_hub_degree) * energy_unit_j
    capacity_j = max(args.energy_budget, recharge_j * 2.0)
    print(f"[energy] recharging power budget: recharge={recharge_j:.1f} J/step "
          f"(sustains hub degree up to {max_hub_degree}), capacity={capacity_j:.1f} J, "
          f"unit={energy_unit_j:.1f} J/link/step; entropy_coef={args.entropy_coef}")

    reports: dict[int, dict] = {}
    for k in args.rounds:
        config = DecentralizedMARLConfig(
            rounds=k,
            hidden_dim=args.hidden_dim,
            critic_hidden_dim=args.hidden_dim,
            rollout_steps=args.rollout_steps,
            minibatch_size=args.minibatch_size,
            scenes_per_update=args.scenes_per_update,
            update_epochs=args.update_epochs,
            max_updates=args.updates,
            warmup_updates=args.warmup_updates,
            bc_teacher_sa_iters=args.bc_sa_iters,
            bc_teacher_restarts=args.bc_restarts,
            eval_every=args.eval_every,
            energy_unit_j=energy_unit_j,
            energy_budget_j=capacity_j,
            energy_recharge_j=recharge_j,
            entropy_coef=args.entropy_coef,
            churn_weight=args.churn_weight,
            num_workers=args.num_workers,
            use_vectorized_evaluator=not args.no_vectorized_evaluator,
            seed=args.seed,
            device=args.device,
        )
        print(f"[train] K={k} ...")
        report, actor, critic = DecentralizedCTDEFlow(config).run(
            train_specs, eval_specs, large_n_specs=large_n_specs
        )
        gate = decentralized_production_gate(report)
        report["production_gate"] = gate
        reports[k] = report
        torch.save(
            {"actor_state_dict": actor.state_dict(),
             "critic_state_dict": critic.state_dict(),
             "config": report["config"]},
            out_dir / f"checkpoint_k{k}.pt",
        )
        (out_dir / f"report_k{k}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        final = report["final_eval"]
        large = report.get("large_n_eval") or {}
        print(f"[train] K={k} eval_feasible={final['tau_feasible_rate']:.3f} "
              f"large_n_feasible={large.get('tau_feasible_rate', float('nan')):.3f} "
              f"gate_passed={gate['passed']}")

    _write_tables(out_dir, reports)
    _write_figures(out_dir, reports)
    _write_report_markdown(out_dir, args, reports)
    print(f"[done] artifacts in {out_dir}")


def _update_series(report: dict, key: str):
    xs, ys = [], []
    for record in report["update_metrics"]:
        if key in record and "update_index" in record:
            xs.append(record["update_index"])
            ys.append(record[key])
    return xs, ys


def _eval_series(report: dict, key: str):
    xs, ys = [], []
    for record in report["eval_metrics"]:
        if key in record:
            xs.append(record["update_index"])
            ys.append(record[key])
    return xs, ys


def _write_tables(out_dir: Path, reports: dict) -> None:
    with (out_dir / "per_update_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rounds_K", "update_index", "train_tau_feasible_rate", "train_mean_reward",
                         "total_loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"])
        for k, report in sorted(reports.items()):
            for record in report["update_metrics"]:
                if "total_loss" not in record:
                    continue
                writer.writerow([k, record.get("update_index"), record.get("train_tau_feasible_rate"),
                                 record.get("train_mean_reward"), record.get("total_loss"),
                                 record.get("policy_loss"), record.get("value_loss"), record.get("entropy"),
                                 record.get("approx_kl"), record.get("clip_fraction")])
    with (out_dir / "final_eval_by_k.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rounds_K", "eval_tau_feasible_rate", "large_n_tau_feasible_rate", "mean_reward",
                         "mean_consensus", "mean_latency", "mean_energy", "mean_churn", "gate_passed"])
        for k, report in sorted(reports.items()):
            final = report["final_eval"]
            large = report.get("large_n_eval") or {}
            gate = report.get("production_gate") or {}
            writer.writerow([k, final["tau_feasible_rate"], large.get("tau_feasible_rate", ""),
                             final["mean_reward"], final["mean_consensus"], final["mean_latency"],
                             final["mean_energy"], final["mean_churn"], gate.get("passed", "")])


def _write_figures(out_dir: Path, reports: dict) -> None:
    plt.figure(figsize=(7, 4.5))
    for k, report in sorted(reports.items()):
        xs, ys = _eval_series(report, "tau_feasible_rate")
        if xs:
            plt.plot(xs, ys, marker="o", label=f"K={k}")
    plt.axhline(0.9, color="grey", linestyle="--", linewidth=0.8, label="target tau=0.9 share")
    plt.xlabel("policy update"); plt.ylabel("held-out per-scene tau-feasible share")
    plt.title("Decentralized MARL: feasibility vs training (K-hop ablation)")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "fig1_feasibility_vs_update.png", dpi=140); plt.close()

    ks = sorted(reports)
    eval_finals = [reports[k]["final_eval"]["tau_feasible_rate"] for k in ks]
    large_finals = [(reports[k].get("large_n_eval") or {}).get("tau_feasible_rate", 0.0) for k in ks]
    x = range(len(ks)); width = 0.38
    plt.figure(figsize=(6.5, 4))
    plt.bar([i - width / 2 for i in x], eval_finals, width, label="in-range N", color="#3b7dd8")
    plt.bar([i + width / 2 for i in x], large_finals, width, label="held-out large N", color="#d8743b")
    plt.xticks(list(x), [str(k) for k in ks])
    plt.xlabel("receptive-field depth K (rounds)"); plt.ylabel("final tau-feasible share")
    plt.title("Multi-hop value & scale generalization by K"); plt.legend()
    plt.tight_layout(); plt.savefig(out_dir / "fig2_final_feasibility_by_k.png", dpi=140); plt.close()

    deepest = max(reports)
    report = reports[deepest]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, key, title in [
        (axes[0, 0], "policy_loss", "policy loss"),
        (axes[0, 1], "value_loss", "value loss"),
        (axes[1, 0], "entropy", "policy entropy"),
        (axes[1, 1], "approx_kl", "approx KL"),
    ]:
        xs, ys = _update_series(report, key)
        if xs:
            ax.plot(xs, ys, marker=".")
        ax.set_title(f"{title} (K={deepest})"); ax.set_xlabel("update")
    fig.tight_layout(); fig.savefig(out_dir / "fig3_training_stability.png", dpi=140); plt.close(fig)


def _write_report_markdown(out_dir: Path, args: argparse.Namespace, reports: dict) -> None:
    designation = production_trunk_designation()
    lines = [
        "# Decentralized CTDE MARL - Training Report",
        "",
        f"- Flow: `{designation['production_trunk_flow_id']}`",
        f"- Actor: `{designation['production_trunk_actor_model_id']}` (K-hop local message passing)",
        f"- Decoder: `{designation['production_trunk_sampler_id']}` (per-node mutual acceptance)",
        f"- Decentralized execution: {designation['production_execution_is_decentralized']}; "
        f"sequential MDP (not bandit): {not designation['production_is_bandit']}",
        f"- Device: {args.device}; scenes: {args.scenario_count} (N in {sorted(args.node_choices)}); "
        f"held-out large N: {sorted(args.large_n)}; updates: {args.updates}; rollout steps: {args.rollout_steps}; "
        f"BC warm-start: {args.warmup_updates}",
        "",
        "## Final results by receptive-field depth K",
        "",
        "| K | in-range tau-feasible | held-out large-N tau-feasible | mean reward | mean consensus | gate passed |",
        "|---|---|---|---|---|---|",
    ]
    for k in sorted(reports):
        f = reports[k]["final_eval"]
        large = reports[k].get("large_n_eval") or {}
        gate = reports[k].get("production_gate") or {}
        lines.append(
            f"| {k} | {f['tau_feasible_rate']:.3f} | {large.get('tau_feasible_rate', float('nan')):.3f} "
            f"| {f['mean_reward']:.3f} | {f['mean_consensus']:.3f} | {gate.get('passed')} |"
        )
    lines += [
        "",
        "## Notes (honest)",
        "- `tau-feasible share` is the fraction of held-out scene-steps whose analytic PBFT consensus",
        "  probability clears tau=0.9 - a per-scene feasibility GRADE, not a single reliability number.",
        "- `held-out large-N` is the scale-generalization test: trained on N in the in-range set, evaluated",
        "  on strictly larger N never seen in training. This is the headline scale result.",
        "- The K-ablation isolates multi-hop value: K=1 ~ ego-graph baseline; larger K reasons over the",
        "  K-hop backbone via local neighbour signalling only.",
        "- Training is BC warm-started on the budget-aware SA teacher, then per-agent CTDE policy gradient.",
        "- Figures: fig1 (learning curve), fig2 (K-ablation + scale generalization), fig3 (training stability).",
        "- Tables: per_update_metrics.csv, final_eval_by_k.csv. Checkpoints: checkpoint_k*.pt.",
    ]
    (out_dir / "TRAINING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
