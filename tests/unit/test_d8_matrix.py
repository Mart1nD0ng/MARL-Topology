"""D8: the 2x2 {motion x recurrence} matrix driver's pure helpers (arm matrix + grouped report)."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_d8():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("d8", root / "scripts" / "diagnostics" / "dynamic_d8_matrix.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_d8_arm_matrix_is_2x2() -> None:
    d8 = _load_d8()
    arms = d8.D8_ARMS
    assert len(arms) == 4
    combos = {(a["actor"] == "recurrent", bool(a["motion"])) for a in arms}
    assert combos == {(False, False), (False, True), (True, False), (True, True)}
    # the velocity arms carry motion; the csi arms do not
    by_label = {a["label"]: a for a in arms}
    assert by_label["memoryless_velocity"]["motion"] is True
    assert by_label["memoryless_csi"]["motion"] is False
    assert by_label["recurrent_csi"]["actor"] == "recurrent"


def test_d8_report_groups_arms_and_baselines() -> None:
    d8 = _load_d8()
    arm_results = {
        "memoryless_csi": [{"held_feas": 0.5, "held_return": -1.0}, {"held_feas": 0.4, "held_return": -1.2}],
        "memoryless_velocity": [{"held_feas": 0.6, "held_return": -0.8}, {"held_feas": 0.7, "held_return": -0.6}],
    }
    deployable = [{"label": "local_threshold", "group": "deployable_policy", "action_evaluator_calls": 0}]
    central = [{"label": "myopic_greedy", "group": "central_reference", "action_evaluator_calls": 50}]
    report = d8.build_report(arm_results, deployable, central, paired={"x": None})
    assert set(report) >= {"learned_arms", "deployable_policies", "central_references", "paired", "scope"}
    # learned arms: per-seed -> CI, never mixed with the baseline groups
    arm = report["learned_arms"]["memoryless_csi"]
    assert arm["n_seeds"] == 2
    assert arm["held_feas"]["n"] == 2 and arm["held_feas"]["mean"] == 0.45
    assert report["deployable_policies"][0]["group"] == "deployable_policy"
    assert report["central_references"][0]["group"] == "central_reference"
    # scope honestly flags single-RSU (not urban)
    assert "SINGLE-RSU" in report["scope"] or "single-RSU" in report["scope"].lower()
