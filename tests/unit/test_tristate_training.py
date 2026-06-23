"""R3 (v2 Engineering-Plan §R3, Spec §5): tri-state solvability controls training.

unknown scenes must ENTER training (D8 -- not deleted via feasible_exists); a finite-search miss
is unknown, never certified_infeasible (#11); train/val/test stay isolated.
"""

from __future__ import annotations

import pytest

from marl_topology.solvability.status import CERTIFIED_INFEASIBLE, UNKNOWN, WITNESS_FEASIBLE
from marl_topology.training.tristate_training import (
    assert_split_isolation,
    solvability_status_for_label,
    trainable_under_tristate,
)


def test_label_status_maps_feasible_exists_to_witness_or_unknown() -> None:
    assert solvability_status_for_label({"feasible_exists": True}) == WITNESS_FEASIBLE
    # a finite-search MISS is UNKNOWN, never certified_infeasible (#11)
    assert solvability_status_for_label({"feasible_exists": False}) == UNKNOWN


def test_label_status_honours_explicit_tristate() -> None:
    assert solvability_status_for_label(
        {"feasible_exists": False, "solvability_status": WITNESS_FEASIBLE}
    ) == WITNESS_FEASIBLE


def test_unknown_trains_by_default_but_can_be_excluded() -> None:
    assert trainable_under_tristate(WITNESS_FEASIBLE) is True
    assert trainable_under_tristate(UNKNOWN) is True                      # D8: unknown trains by default
    assert trainable_under_tristate(UNKNOWN, exclude_unknown=True) is False  # ablation opt-out only
    # witness_feasible always trains, even under the unknown ablation
    assert trainable_under_tristate(WITNESS_FEASIBLE, exclude_unknown=True) is True


def test_certified_infeasible_never_trains() -> None:
    assert trainable_under_tristate(CERTIFIED_INFEASIBLE) is False
    assert trainable_under_tristate(CERTIFIED_INFEASIBLE, exclude_unknown=True) is False


def test_unknown_label_is_not_filtered_out_like_the_old_binary_gate() -> None:
    # the OLD trunk did `if not feasible_exists: continue` -> skipped this scene. R3 must train it.
    unknown_label = {"feasible_exists": False}
    assert trainable_under_tristate(solvability_status_for_label(unknown_label)) is True


def test_bad_status_raises() -> None:
    with pytest.raises(ValueError):
        trainable_under_tristate("infeasible")  # not a tri-state member


def test_split_isolation_assertion() -> None:
    assert_split_isolation({"s1", "s2"}, {"s3"}, {"s4", "s5"})  # disjoint -> ok
    with pytest.raises(AssertionError, match="isolation violated"):
        assert_split_isolation({"s1", "s2"}, {"s2", "s3"})  # s2 in two splits
