"""Phase 1a: safe, configurable PBFT quorum spec (classic_exact / safe_generalized).

These tests pin the quorum-intersection (safety) and liveness conditions the
MARL-Topology-Technical-Spec.md (S4.3) requires, and the regression that the old
hardcoded q = 2f + 1 is UNSAFE for the generalized n > 3f + 1 the project runs.
"""

from __future__ import annotations

import pytest

from marl_topology.protocol import (
    QUORUM_MODE_CLASSIC,
    QUORUM_MODE_SAFE,
    PBFTExpectedInitiatorConfig,
    PBFTQuorumSpec,
    PBFTThreePhaseConfig,
)


def _safe_q(n: int, f: int) -> int:
    return (n + f) // 2 + 1


# --- the safety + liveness conditions hold by construction (safe_generalized) ---

def test_safe_generalized_quorum_formula() -> None:
    for n in range(4, 40):
        for f in range(0, (n - 1) // 3 + 1):
            spec = PBFTQuorumSpec(node_count=n, fault_tolerance=f, mode=QUORUM_MODE_SAFE)
            assert spec.quorum == _safe_q(n, f)
            assert spec.external_quorum == spec.quorum - 1


def test_safe_generalized_satisfies_intersection_and_liveness() -> None:
    for n in range(4, 60):
        for f in range(0, (n - 1) // 3 + 1):
            spec = PBFTQuorumSpec(node_count=n, fault_tolerance=f, mode=QUORUM_MODE_SAFE)
            q = spec.quorum
            # quorum intersection contains >= 1 honest node
            assert 2 * q - n > f, (n, f, q)
            # the n - f honest nodes can still form a quorum (liveness)
            assert q <= n - f, (n, f, q)


def test_classic_equals_safe_at_n_equals_3f_plus_1() -> None:
    for f in range(0, 8):
        n = 3 * f + 1
        classic = PBFTQuorumSpec(node_count=n, fault_tolerance=f, mode=QUORUM_MODE_CLASSIC)
        safe = PBFTQuorumSpec(node_count=n, fault_tolerance=f, mode=QUORUM_MODE_SAFE)
        assert classic.quorum == 2 * f + 1
        assert classic.quorum == safe.quorum
        assert classic.external_quorum == safe.external_quorum == 2 * f


def test_classic_exact_rejects_generalized_n() -> None:
    # n = 8, f = 2 is a valid PBFT committee (8 >= 3*2+1) but n != 3f+1.
    with pytest.raises(ValueError, match="classic_exact requires n == 3f"):
        PBFTQuorumSpec(node_count=8, fault_tolerance=2, mode=QUORUM_MODE_CLASSIC)


def test_rejects_n_below_3f_plus_1() -> None:
    for mode in (QUORUM_MODE_CLASSIC, QUORUM_MODE_SAFE):
        with pytest.raises(ValueError, match="n >= 3f"):
            PBFTQuorumSpec(node_count=6, fault_tolerance=2, mode=mode)


def test_the_old_2f_plus_1_is_unsafe_for_n_gt_3f_plus_1() -> None:
    """Regression pinning the Phase-1 bug: the naive classic q = 2f+1, applied at the
    generalized n the project actually runs, violates quorum intersection."""
    n, f = 8, 2
    naive_q = 2 * f + 1  # = 5, the old PBFTThreePhaseConfig.total_quorum
    assert 2 * naive_q - n == 2  # intersection = 2 honest-or-not nodes...
    assert not (2 * naive_q - n > f)  # ...which is NOT strictly greater than f -> UNSAFE
    safe = PBFTQuorumSpec(node_count=n, fault_tolerance=f, mode=QUORUM_MODE_SAFE)
    assert safe.quorum == 6
    assert 2 * safe.quorum - n > f  # the fix restores intersection safety


# --- the production configs now derive their quorum from the safe spec ---

def test_three_phase_config_uses_safe_quorum_for_generalized_n() -> None:
    nodes = tuple(f"n{i}" for i in range(8))
    cfg = PBFTThreePhaseConfig(node_ids=nodes, primary_id="n0", fault_tolerance=2)
    # default mode is safe_generalized -> q = 6, external = 5 (was 5 / 4 under the bug)
    assert cfg.total_quorum == 6
    assert cfg.external_quorum == 5


def test_classic_n_equals_3f_plus_1_unchanged() -> None:
    nodes = ("n0", "n1", "n2", "n3")
    cfg = PBFTThreePhaseConfig(node_ids=nodes, primary_id="n0", fault_tolerance=1)
    assert cfg.total_quorum == 3
    assert cfg.external_quorum == 2


def test_expected_initiator_config_exposes_safe_quorum() -> None:
    nodes = tuple(f"n{i}" for i in range(12))
    cfg = PBFTExpectedInitiatorConfig(node_ids=nodes, fault_tolerance=3)
    # n=12, f=3: safe q = (12+3)//2 + 1 = 8 (vs the unsafe 2f+1 = 7)
    assert cfg.quorum_spec.quorum == 8
    assert 2 * cfg.quorum_spec.quorum - 12 > 3
