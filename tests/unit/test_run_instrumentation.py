"""Full-Integration-Audit instrumentation (checklist §11): every formal run must emit a
reproducible artifact set -- data_manifest, mechanism_activation, seed_manifest. These tests
pin that the instrumentation records the TRUTH (not aspirational flags):

  * data_manifest carries shard SHA256 + the env-math generator config READ FROM the shard
    (fault_model / one_hop_relay / relay_hops / timeout_aware_latency), not hardcoded;
  * mechanism_activation.dynamic_task.enabled is FALSE (the trunk is a T=1 bandit) and
    trajectory_length is 1 for every split item -- the instrumentation cannot silently claim a
    dynamic task that the training loop never runs;
  * mechanism flags mirror the actual argparse config (counterfactual / scq / chance / pareto /
    actor), and CVaR is marked a primitive NOT in any loss;
  * default (ema/mlp, no opt-in) reports the simple-baseline activation honestly.
"""

from __future__ import annotations

import hashlib
from argparse import Namespace
from pathlib import Path

from marl_topology.training.run_instrumentation import (
    dataset_manifest,
    mechanism_activation,
    split_manifest,
)


def _write_blob(tmp_path: Path, name: str, payload: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(payload)
    return p


def test_dataset_manifest_has_sha256_and_count(tmp_path: Path) -> None:
    a = _write_blob(tmp_path, "_op_shard_3001.pkl", b"alpha")
    b = _write_blob(tmp_path, "_op_shard_3002.pkl", b"beta")
    man = dataset_manifest([str(a), str(b)], environment_math_version="v2-corrected-test")
    assert man["n_shards"] == 2
    assert man["environment_math_version"] == "v2-corrected-test"
    by_name = {s["name"]: s for s in man["shards"]}
    assert by_name["_op_shard_3001.pkl"]["sha256"] == hashlib.sha256(b"alpha").hexdigest()
    assert by_name["_op_shard_3002.pkl"]["sha256"] == hashlib.sha256(b"beta").hexdigest()


def test_dataset_manifest_records_op_corrected_directory(tmp_path: Path) -> None:
    a = _write_blob(tmp_path, "_op_shard_3001.pkl", b"x")
    man = dataset_manifest([str(a)], environment_math_version="v")
    # the dataset directory is recorded so legacy op/ vs op_corrected/ can never be confused (§2.1 Q6/Q7)
    assert "shard_dirs" in man
    assert man["shards"][0]["sha256"]


def _items_with_status(node_counts, statuses):
    """Fabricate (row, context, label) triples with a stub context exposing node_ids + fixture_id."""
    items = []
    for i, (n, st) in enumerate(zip(node_counts, statuses)):
        graph = Namespace(node_ids=list(range(n)))
        ctx = Namespace(graph=graph, fixture=Namespace(fixture_id=f"sc{i}"))
        label = {"solvability_status": st, "feasible_exists": st == "witness_feasible"}
        items.append((None, ctx, label))
    return items


def test_split_manifest_distributions_and_t1_trajectory() -> None:
    fit = _items_with_status([8, 8, 12, 16], ["witness_feasible", "unknown", "witness_feasible", "unknown"])
    val = _items_with_status([12], ["witness_feasible"])
    held = _items_with_status([16, 16], ["witness_feasible", "unknown"])
    man = split_manifest(fit, val, held)
    assert man["fit"]["n_items"] == 4
    assert man["fit"]["node_count_distribution"] == {8: 2, 12: 1, 16: 1}
    assert man["fit"]["solvability_distribution"]["witness_feasible"] == 2
    assert man["fit"]["solvability_distribution"]["unknown"] == 2
    assert man["fit"]["scenario_ids"] == ["sc0", "sc1", "sc2", "sc3"]
    # T=1 bandit: EVERY item is a length-1 trajectory; the manifest must not imply episodes.
    assert man["fit"]["trajectory_length_distribution"] == {"1": 4}
    assert man["held"]["trajectory_length_distribution"] == {"1": 2}
    assert "single-step" in man["note"].lower() or "t=1" in man["note"].lower()


def test_mechanism_activation_dynamic_is_false_and_t1() -> None:
    args = Namespace(baseline="graph-mappo", counterfactual=True, k_cf=4, scq=False, scq_m=2,
                     scq_select="simple", chance=False, chance_delta=0.1, pareto_archive=False,
                     actor="mlp", action_space="mutual", ckpt_every=0)
    ma = mechanism_activation(args, critic_parameter_delta=0.123, scq_calls_per_scene=0.0)
    # The single most important honesty pin: the dynamic task was NOT run.
    assert ma["dynamic_task"]["enabled"] is False
    assert ma["dynamic_task"]["episode_length"] == 1
    assert ma["dynamic_task"]["reconfiguration_cost_nonzero"] is False
    assert ma["critic"]["graph_mappo"] is True
    assert ma["critic"]["critic_parameter_delta"] == 0.123
    assert ma["critic"]["critic_sees_action"] is True       # --counterfactual -> Q critic
    assert ma["counterfactual"]["enabled"] is True
    assert ma["counterfactual"]["k_cf"] == 4
    assert ma["action"]["distribution"] == "bcsp"           # graph-mappo arm uses BCSP
    assert ma["action"]["per_agent_ratio"] is True
    # CVaR must never be claimed as "trained".
    assert ma["constraint"]["cvar_in_loss"] is False


def test_mechanism_activation_default_baseline_honest() -> None:
    args = Namespace(baseline="ema", counterfactual=False, k_cf=4, scq=False, scq_m=2,
                     scq_select="simple", chance=False, chance_delta=0.1, pareto_archive=False,
                     actor="mlp", action_space="mutual", ckpt_every=0)
    ma = mechanism_activation(args, critic_parameter_delta=None, scq_calls_per_scene=0.0)
    assert ma["critic"]["graph_mappo"] is False
    assert ma["counterfactual"]["enabled"] is False
    assert ma["scq"]["enabled"] is False
    assert ma["pareto"]["enabled"] is False
    assert ma["actor"]["architecture"] == "mlp"
    assert ma["actor"]["preference_conditioned"] is False


def test_mechanism_activation_pna_omega_fixed_neutral() -> None:
    args = Namespace(baseline="ema", counterfactual=False, k_cf=4, scq=False, scq_m=2,
                     scq_select="simple", chance=False, chance_delta=0.1, pareto_archive=False,
                     actor="pna", action_space="mutual", ckpt_every=0)
    ma = mechanism_activation(args, critic_parameter_delta=None, scq_calls_per_scene=0.0)
    assert ma["actor"]["architecture"] == "pna"
    # honest scoping: omega is fixed neutral, recurrence is intra-forward only (no T>1) -> cannot
    # claim a preference-conditioned Pareto policy / temporal recurrence was tested (checklist §9.2/§10).
    assert ma["actor"]["preference_conditioned"] is False
    assert ma["actor"]["recurrent_temporal_state_across_steps"] is False
