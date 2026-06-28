"""Q12 (POMDP-QP-FAR): the consolidated campaign helpers.

Pins the per-seed 95% CI helper and the deployable-vs-central grouping (which asserts the deployable arms
make 0 action-evaluator calls). Fails on HEAD (the module is new).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts" / "diagnostics"))
sys.path.insert(0, str(_ROOT / "scripts" / "train"))

import pomdp_qpfar_campaign as q12  # noqa: E402


def test_campaign_paired_ci_helper() -> None:
    z = q12.ci95([0.5, 0.5, 0.5, 0.5, 0.5])
    assert z["mean"] == pytest.approx(0.5) and z["lo"] == pytest.approx(0.5) and z["hi"] == pytest.approx(0.5)
    c = q12.ci95([1.0, 2.0, 3.0, 4.0, 5.0])                       # mean 3, sd 1.5811, t(5)=2.776 -> h~1.963
    assert c["mean"] == pytest.approx(3.0, abs=1e-6)
    assert c["lo"] == pytest.approx(3.0 - 1.9628, abs=1e-3) and c["hi"] == pytest.approx(3.0 + 1.9628, abs=1e-3)
    assert q12.ci95([])["n"] == 0


def test_campaign_groups_deployable_vs_central() -> None:
    arms = [
        {"label": "local_hysteresis(anchor)", "group": "deployable_policy", "action_evaluator_calls": 0},
        {"label": "local_threshold", "group": "deployable_policy", "action_evaluator_calls": 0},
        {"label": "myopic_greedy(oracle)", "group": "central_reference", "action_evaluator_calls": 864},
    ]
    g = q12.group_arms(arms)
    assert set(g["groups"]) == {"deployable_policy", "central_reference"}
    assert len(g["groups"]["deployable_policy"]) == 2 and len(g["groups"]["central_reference"]) == 1
    assert g["deployable_use_no_evaluator_for_action"] is True     # deployable arms make 0 eval calls
    # a mislabeled deployable arm (with eval calls) must trip the assertion
    bad = arms + [{"label": "x", "group": "deployable_policy", "action_evaluator_calls": 5}]
    assert q12.group_arms(bad)["deployable_use_no_evaluator_for_action"] is False
