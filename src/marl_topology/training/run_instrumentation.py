"""Full-Integration-Audit run instrumentation (audit checklist §11 / §1 minimal-evidence-chain).

Every formal training run should emit a reproducible, *honest* artifact set so a reviewer can
confirm WHAT was actually activated -- not what is merely implemented. This module builds the
three provenance artifacts the checklist demands and that the trunk previously did not write:

  * ``data_manifest``     -- shard filenames + SHA256, the env-math generator config READ FROM the
                             shard (not hardcoded), and the dataset directory (op vs op_corrected);
  * ``split_manifest``    -- per-split scenario IDs + node-count / solvability / trajectory-length
                             distributions. trajectory_length is 1 for every item: the trunk is a
                             single-step (T=1) contextual bandit -- the manifest cannot imply a
                             dynamic task the loop never runs;
  * ``mechanism_activation`` -- the activated-mechanism map (D6), mirroring the actual argparse
                             config + the MEASURED critic-parameter delta + SCQ evaluator budget.
                             dynamic_task.enabled is FALSE; CVaR is flagged a primitive-not-in-loss;
                             the PNA actor's omega is flagged fixed-neutral (no preference sweep).

Lives under ``training/`` (gate-exempt); imported only by training entrypoints, never deployed.
"""

from __future__ import annotations

import hashlib
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

from .run_manifest import ENVIRONMENT_MATH_VERSION
from .tristate_training import solvability_status_for_label

# env-math fields stamped on the dataset-generator's PhysicsRegime (read, never assumed).
_REGIME_ENV_MATH_FIELDS = (
    "fault_model", "one_hop_relay", "relay_hops", "timeout_aware_latency",
    "scheduled_mac", "coverage_gated_membership", "path_loss_model",
)


def sha256_of(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shard_generator_config(shard_path: str | Path) -> dict[str, Any]:
    """Best-effort read of a shard's generator config + the env-math regime fields it was built
    with. Returns ``{}`` if the shard cannot be opened (manifest still records the SHA256)."""
    try:
        with open(shard_path, "rb") as handle:
            dataset = pickle.load(handle)
    except Exception:
        return {}
    cfg = getattr(dataset, "config", None)
    out: dict[str, Any] = {"dataset_id": getattr(dataset, "dataset_id", None)}
    if cfg is not None:
        for attr in ("seed", "scenario_count", "node_count_choices", "tau_requirement_min",
                     "target_feasible_fraction", "target_near_threshold_fraction",
                     "target_infeasible_fraction", "vectorized_evaluator"):
            if hasattr(cfg, attr):
                out[attr] = getattr(cfg, attr)
        regime = getattr(cfg, "regime", None)
        if regime is not None:
            out["env_math_regime"] = {f: getattr(regime, f) for f in _REGIME_ENV_MATH_FIELDS
                                      if hasattr(regime, f)}
    return out


def dataset_manifest(shard_paths: Sequence[str | Path],
                     environment_math_version: str = ENVIRONMENT_MATH_VERSION) -> dict[str, Any]:
    """Shard SHA256 + dataset directory + the env-math generator config read from the first shard."""
    shards = []
    dirs = []
    for p in shard_paths:
        pp = Path(p)
        shards.append({"name": pp.name, "path": str(p), "sha256": sha256_of(pp)})
        d = str(pp.parent)
        if d not in dirs:
            dirs.append(d)
    gen_cfg = shard_generator_config(shard_paths[0]) if shard_paths else {}
    return {
        "environment_math_version": environment_math_version,
        "n_shards": len(shards),
        "shard_dirs": dirs,
        "generator_config": gen_cfg,            # incl. env_math_regime (fault_model/relay/timeout)
        "shards": shards,
    }


def _node_count(item) -> int:
    return len(item[1].graph.node_ids)


def _scenario_id(item):
    return getattr(getattr(item[1], "fixture", None), "fixture_id", None)


def _status(item, status_of: Callable[[Any], str]) -> str:
    return status_of(item[2])


def split_manifest(fit_items, val_items, held_items,
                   status_of: Callable[[Any], str] = solvability_status_for_label) -> dict[str, Any]:
    """Per-split scenario IDs + node-count / solvability / trajectory-length distributions.

    trajectory_length is 1 for EVERY item -- the trunk is a single-step (T=1) contextual bandit;
    episodes / topology persistence / reconfiguration are NOT modeled (the dynamic two-timescale
    env is opt-in and unwired). The note makes that scope explicit so the manifest cannot be read
    as covering a dynamic task.
    """
    def block(items):
        ncounts = Counter(_node_count(it) for it in items)
        solv = Counter(_status(it, status_of) for it in items)
        return {
            "n_items": len(items),
            "scenario_ids": [_scenario_id(it) for it in items],
            "node_count_distribution": {int(k): int(v) for k, v in sorted(ncounts.items())},
            "solvability_distribution": {str(k): int(v) for k, v in sorted(solv.items())},
            "trajectory_length_distribution": {"1": len(items)},
        }
    return {
        "fit": block(fit_items),
        "val": block(val_items),
        "held": block(held_items),
        "note": ("trajectory_length is 1 for ALL items: the formal training entry is a single-step "
                 "(T=1) contextual bandit. Episodes / topology persistence / reconfiguration cost / "
                 "return-over-horizon are NOT modeled here (two_timescale_env.py is opt-in and "
                 "unwired). No dynamic-task conclusion is licensed by this run."),
    }


def _action_distribution(args) -> str:
    if getattr(args, "baseline", None) == "graph-mappo":
        return "bcsp"                                   # R6/R7 unordered subset policy
    space = getattr(args, "action_space", "mutual")
    return {"mutual": "plackett_luce", "gauss": "gaussian_logit", "bernoulli": "bernoulli"}.get(space, space)


def mechanism_activation(args, *, critic_parameter_delta: float | None = None,
                         scq_calls_per_scene: float = 0.0,
                         dataset_env_math: dict | None = None) -> dict[str, Any]:
    """The activated-mechanism map (D6), mirroring the actual argparse config + measured signals.

    HONESTY INVARIANTS (pinned by tests): dynamic_task.enabled is always False (T=1 bandit);
    CVaR is flagged not-in-loss; the PNA actor's omega is flagged fixed-neutral (no sweep) and its
    recurrence intra-forward only (no temporal hidden-state across episode steps, since T=1).
    """
    is_gm = getattr(args, "baseline", None) == "graph-mappo"
    is_cf = bool(getattr(args, "counterfactual", False))
    is_pna = getattr(args, "actor", "mlp") == "pna"
    env_math = {"version": ENVIRONMENT_MATH_VERSION}
    if dataset_env_math:
        env_math.update(dataset_env_math)              # fault_model / one_hop_relay / relay_hops / ...
    return {
        "dynamic_task": {
            "enabled": False,
            "episode_length": 1,
            "hold_interval": None,
            "reconfiguration_cost_nonzero": False,
            "note": "T=1 contextual bandit; two_timescale_env.py exists but is NOT wired into this entry.",
        },
        "environment_math": env_math,
        "action": {
            "distribution": _action_distribution(args),
            "per_agent_ratio": is_gm,                  # per-agent PPO ratio lives in the graph-mappo arm
            "local_mutual_decoder": True,              # the unique train==deploy decoder
        },
        "critic": {
            "graph_mappo": is_gm,
            "critic_sees_action": is_cf,               # action-conditioned Q critic iff --counterfactual
            "critic_parameter_delta": critic_parameter_delta,
            "critic_checkpointed": bool(getattr(args, "ckpt_every", 0)),
        },
        "counterfactual": {
            "enabled": is_cf,
            "k_cf": getattr(args, "k_cf", 0) if is_cf else 0,
            "per_agent_advantage": is_cf,
        },
        "scq": {
            "enabled": bool(getattr(args, "scq", False)),
            "scq_m": getattr(args, "scq_m", 0),
            "select": getattr(args, "scq_select", None),
            "evaluator_calls_per_scene": scq_calls_per_scene,
        },
        "constraint": {
            "chance_enabled": bool(getattr(args, "chance", False)),
            "chance_delta": getattr(args, "chance_delta", None),
            "cvar_in_loss": False,
            "cvar_note": "cvar_shortfall is a verified primitive; NOT wired into any training loss or "
                         "checkpoint selection (do not report as 'CVaR trained').",
        },
        "pareto": {
            "enabled": bool(getattr(args, "pareto_archive", False)),
            "selected_by_archive": bool(getattr(args, "pareto_archive", False)),
        },
        "actor": {
            "architecture": getattr(args, "actor", "mlp"),
            "preference_conditioned": False,           # omega is fixed neutral; never swept in train/eval
            "omega": "fixed_neutral_(0.5,0.5)" if is_pna else "n/a",
            "recurrent_temporal_state_across_steps": False,
            "note": ("omega never swept and (T=1) no temporal hidden-state across episode steps -- the "
                     "PNA actor's recurrence is intra-forward message rounds only." if is_pna else None),
        },
    }
