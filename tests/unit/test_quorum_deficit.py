"""Q3 (POMDP-QP-FAR): quorum-shortfall feasibility distance D_quorum (Spec S8).

Pins: (a) the Poisson-binomial low-pmf + expected shortfall against brute-force enumeration; (b) the
grounding identity that the deficit's tail bucket IS the reliability's quorum tail; (c) the spec's
qualitative properties (zero when quorum met, high when no messages, monotone-decreasing in message
prob); (d) the per-phase/per-receiver cascade structure; and (e) the D_quorum vs true-C anti-correlation
premise that Q4 will test rigorously. Fails on HEAD (the module is new).
"""

from __future__ import annotations

import itertools
import math

import pytest

from marl_topology.protocol.quorum_deficit import (
    deficit_cvar,
    expected_initiator_quorum_deficit,
    expected_quorum_shortfall,
    per_primary_quorum_deficit,
    poisson_binomial_low_pmf,
)
from marl_topology.protocol.quorum_tail import heterogeneous_quorum_tail
from marl_topology.protocol.pbft_reliability import (
    PBFTExpectedInitiatorConfig,
    evaluate_expected_initiator_pbft_reliability,
)


def _bruteforce_pmf(probs):
    n = len(probs)
    pmf = [0.0] * (n + 1)
    for bits in itertools.product((0, 1), repeat=n):
        p, s = 1.0, 0
        for b, pi in zip(bits, probs):
            p *= pi if b else (1.0 - pi)
            s += b
        pmf[s] += p
    return pmf


def _matrix(node_ids, prob):
    return {(a, b): prob for a in node_ids for b in node_ids if a != b}


# -- pure primitives vs brute force -----------------------------------------------------------------

def test_low_pmf_matches_bruteforce() -> None:
    probs = [0.2, 0.5, 0.9, 0.4, 0.7]
    q = 3
    low = poisson_binomial_low_pmf(probs, q)
    bf = _bruteforce_pmf(probs)
    for k in range(q):
        assert low[k] == pytest.approx(bf[k], abs=1e-12)
    assert low[q] == pytest.approx(sum(bf[k] for k in range(q, len(bf))), abs=1e-12)   # tail bucket


def test_tail_bucket_equals_quorum_tail() -> None:
    # The grounding identity: D_quorum shares the reliability's exact Poisson-binomial.
    probs = [0.1, 0.3, 0.55, 0.8, 0.95, 0.2]
    for q in range(0, len(probs) + 2):
        low = poisson_binomial_low_pmf(probs, q)
        assert low[-1] == pytest.approx(heterogeneous_quorum_tail(probs, q), abs=1e-12)


def test_expected_shortfall_matches_bruteforce() -> None:
    probs = [0.3, 0.6, 0.2, 0.8]
    for q in range(0, 6):
        bf = sum(max(0, q - k) * pk for k, pk in enumerate(_bruteforce_pmf(probs)))
        assert expected_quorum_shortfall(probs, q) == pytest.approx(bf, abs=1e-12)


def test_shortfall_zero_when_quorum_always_met() -> None:
    assert expected_quorum_shortfall([1.0, 1.0, 1.0, 1.0], 3) == pytest.approx(0.0, abs=1e-12)
    assert expected_quorum_shortfall([0.5, 0.5], 0) == 0.0


def test_shortfall_high_when_no_messages() -> None:
    assert expected_quorum_shortfall([0.0, 0.0, 0.0], 3) == pytest.approx(3.0, abs=1e-12)  # short by all q
    assert expected_quorum_shortfall([0.0, 0.0], 5) == pytest.approx(5.0, abs=1e-12)       # q > n


def test_shortfall_decreases_when_message_probability_increases() -> None:
    last = None
    for p in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        d = expected_quorum_shortfall([p, p, p, p, p], 3)
        if last is not None:
            assert d < last - 1e-9
        last = d


def test_deficit_cvar_is_upper_tail_mean() -> None:
    vals = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]   # 10 vals, worst 10% = the 2.0
    assert deficit_cvar(vals, 0.9) == pytest.approx(2.0, abs=1e-12)
    assert deficit_cvar([], 0.9) == 0.0
    # float ceil edge: (1-0.7)*10 = 3.0000..04 must NOT round up to 4 -> worst 3 of 10 = mean(3,2,1)=2.0
    assert deficit_cvar([0.0] * 7 + [1.0, 2.0, 3.0], 0.7) == pytest.approx(2.0, abs=1e-12)


# -- per-phase/receiver structure + grounding to reliability ----------------------------------------

def test_deficit_uses_phase_receiver_structure() -> None:
    ids = ("v0", "v1", "v2", "v3")
    summ = per_primary_quorum_deficit(ids, "v0", 1, pre_prepare_matrix=_matrix(ids, 0.6),
                                      prepare_matrix=_matrix(ids, 0.6), commit_matrix=_matrix(ids, 0.6))
    phases = {ph for (ph, _r) in summ.per_phase_receiver}
    assert "prepare" in phases and "commit" in phases and ("global", "*") in summ.per_phase_receiver
    assert {r for (_p, r) in summ.per_phase_receiver if _p == "prepare"} == set(ids)   # one term per receiver
    assert summ.d_max >= summ.d_mean >= 0.0


def test_perfect_delivery_zero_deficit_full_reliability() -> None:
    ids = ("v0", "v1", "v2", "v3")
    summ = per_primary_quorum_deficit(ids, "v0", 1, pre_prepare_matrix=_matrix(ids, 1.0),
                                      prepare_matrix=_matrix(ids, 1.0), commit_matrix=_matrix(ids, 1.0))
    assert summ.d_max == pytest.approx(0.0, abs=1e-9)        # everyone meets quorum -> no shortfall
    rel = evaluate_expected_initiator_pbft_reliability(
        PBFTExpectedInitiatorConfig(node_ids=ids, fault_tolerance=1),
        pre_prepare_matrix=_matrix(ids, 1.0), prepare_matrix=_matrix(ids, 1.0), commit_matrix=_matrix(ids, 1.0))
    assert rel.consensus_success_probability == pytest.approx(1.0, abs=1e-9)


def test_deficit_has_gradient_where_reliability_is_flat() -> None:
    # THE motivation (Spec S7.1/S8): on the sub-feasible plateau the true C is FLAT at 0, but D_quorum
    # still decreases as the channel improves -> a usable gradient where C has none. Previewed here on
    # synthetic matrices; Q4 tests it rigorously under local edits on real topologies.
    ids = ("v0", "v1", "v2", "v3", "v4")
    seq = []
    for p in (0.5, 0.7, 0.9, 0.99, 0.999):
        c = evaluate_expected_initiator_pbft_reliability(
            PBFTExpectedInitiatorConfig(node_ids=ids, fault_tolerance=1),
            pre_prepare_matrix=_matrix(ids, p), prepare_matrix=_matrix(ids, p),
            commit_matrix=_matrix(ids, p)).consensus_success_probability
        d = expected_initiator_quorum_deficit(
            ids, 1, pre_prepare_matrix=_matrix(ids, p), prepare_matrix=_matrix(ids, p),
            commit_matrix=_matrix(ids, p))["d_quorum_mean"]
        seq.append((p, c, d))
    for (_, _, d_prev), (_, _, d_cur) in zip(seq, seq[1:]):
        assert d_cur < d_prev - 1e-6                          # D strictly decreases as the channel improves
    for (_, c_prev, _), (_, c_cur, _) in zip(seq, seq[1:]):
        assert c_cur >= c_prev - 1e-9                         # C is non-decreasing
    plateau = [(c, d) for (_, c, d) in seq if c < 1e-3]       # the sub-feasible plateau (C ~ 0, << tau=0.9)
    assert len(plateau) >= 2 and plateau[0][1] > plateau[-1][1] + 0.5   # D moved while C stayed flat at ~0
    assert seq[-1][2] < 0.05                                  # near-perfect channel -> near-zero deficit
