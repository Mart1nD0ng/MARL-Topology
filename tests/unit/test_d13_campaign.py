"""D13: the full-campaign driver's pure helpers (arm spec + per-data report grouping). Heavy subprocess
runs are exercised by the pilot/background run, not here."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "d13", ROOT / "scripts" / "diagnostics" / "dynamic_d13_campaign.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_arm_spec_covers_urban_and_random_and_single_variable() -> None:
    d13 = _load()
    datas = {a["data"] for a in d13.D13_ARMS}
    assert datas == {"urban", "random"}, "campaign must contrast urban vs random"
    labels = {(a["data"], a["label"]) for a in d13.D13_ARMS}
    assert ("urban", "baseline") in labels and ("random", "baseline") in labels  # the contrast point
    # the D9-D12 mechanism A/Bs all present on urban
    for mech in ("coma", "scq", "chance", "pareto", "pna"):
        assert ("urban", mech) in labels, f"missing mechanism arm {mech}"
    # each mechanism arm is single-variable vs baseline: pna swaps arch, coma/scq/chance/pareto add flags
    coma = next(a for a in d13.D13_ARMS if a["label"] == "coma")
    assert "--counterfactual" in coma["flags"] and coma["arch"] == "mlp"
    pna = next(a for a in d13.D13_ARMS if a["label"] == "pna")
    assert pna["arch"] == "pna"


def test_ci_reports_per_seed_and_interval() -> None:
    d13 = _load()
    ci = d13._ci([0.5, 0.6, 0.7, 0.8, 0.9])
    assert ci["n"] == 5 and ci["seeds"] == [0.5, 0.6, 0.7, 0.8, 0.9]
    assert ci["ci95"][0] < ci["mean"] < ci["ci95"][1]   # a real interval (>=2 seeds)
    assert d13._ci([]) is None


def test_build_report_groups_per_data_never_mixes() -> None:
    d13 = _load()
    arm_results = {
        ("urban", "baseline"): [{"held_feas": 0.3}, {"held_feas": 0.4}],
        ("urban", "coma"): [{"held_feas": 0.35}, {"held_feas": 0.45}],
        ("random", "baseline"): [{"held_feas": 0.6}, {"held_feas": 0.7}],
    }
    baselines = {
        "urban": {"deployable_policies": [{"label": "local_threshold", "group": "deployable_policy"}],
                  "central_references": [{"label": "myopic_greedy", "group": "central_reference"}]},
        "random": {"deployable_policies": [{"label": "local_threshold", "group": "deployable_policy"}],
                   "central_references": [{"label": "myopic_greedy", "group": "central_reference"}]},
    }
    paired = {"coma_minus_baseline_feas": d13._ci([0.05, 0.05])}
    rep = d13.build_report(arm_results, baselines, paired)
    # learned arms grouped under their data; baselines separate; central never under learned
    assert set(rep["by_data"]) == {"urban", "random"}
    assert "baseline" in rep["by_data"]["urban"]["learned_arms"]
    assert "coma" in rep["by_data"]["urban"]["learned_arms"]
    assert "coma" not in rep["by_data"]["random"]["learned_arms"]   # no cross-data leak
    assert rep["by_data"]["urban"]["central_references"][0]["group"] == "central_reference"
    assert rep["by_data"]["urban"]["learned_arms"]["baseline"]["held_feas"]["n"] == 2
    assert "coma_minus_baseline_feas" in rep["paired_vs_urban_baseline"]
