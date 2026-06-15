from pathlib import Path

import pytest

from marl_topology.protocol import (
    QUORUM_TAIL_EVALUATOR_ID,
    conservative_quorum_tail,
    evaluate_quorum_tail,
    heterogeneous_quorum_tail,
    remove_largest_probabilities,
)


ROOT = Path(__file__).resolve().parents[2]


def test_quorum_tail_boundary_cases() -> None:
    assert heterogeneous_quorum_tail([], 0) == 1.0
    assert heterogeneous_quorum_tail([], 1) == 0.0
    assert heterogeneous_quorum_tail([0.2, 0.3], 0) == 1.0
    assert heterogeneous_quorum_tail([0.2, 0.3], 3) == 0.0
    assert heterogeneous_quorum_tail([0.0, 0.0, 0.0], 1) == 0.0
    assert heterogeneous_quorum_tail([1.0, 1.0, 1.0], 3) == 1.0


def test_quorum_tail_matches_homogeneous_hand_formulas() -> None:
    p = 0.25
    probabilities = [p, p, p]

    at_least_one = 1.0 - (1.0 - p) ** 3
    at_least_two = 3.0 * p * p * (1.0 - p) + p**3
    at_least_three = p**3

    assert heterogeneous_quorum_tail(probabilities, 1) == pytest.approx(at_least_one)
    assert heterogeneous_quorum_tail(probabilities, 2) == pytest.approx(at_least_two)
    assert heterogeneous_quorum_tail(probabilities, 3) == pytest.approx(at_least_three)


def test_quorum_tail_matches_heterogeneous_small_hand_formula() -> None:
    p1, p2, p3 = 0.2, 0.5, 0.8
    expected_at_least_two = (
        p1 * p2 * (1.0 - p3)
        + p1 * p3 * (1.0 - p2)
        + p2 * p3 * (1.0 - p1)
        + p1 * p2 * p3
    )

    assert heterogeneous_quorum_tail([p1, p2, p3], 2) == pytest.approx(
        expected_at_least_two
    )


def test_quorum_tail_is_monotonic_in_each_probability() -> None:
    base = [0.2, 0.4, 0.6, 0.8]
    base_value = heterogeneous_quorum_tail(base, 3)

    for index in range(len(base)):
        improved = list(base)
        improved[index] = min(1.0, improved[index] + 0.1)
        assert heterogeneous_quorum_tail(improved, 3) >= base_value


def test_quorum_tail_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="probabilities must be in"):
        heterogeneous_quorum_tail([1.2], 1)
    with pytest.raises(ValueError, match="probabilities must be finite"):
        heterogeneous_quorum_tail([float("nan")], 1)
    with pytest.raises(ValueError, match="quorum_size must be nonnegative"):
        heterogeneous_quorum_tail([0.5], -1)
    with pytest.raises(TypeError, match="quorum_size must be an integer"):
        heterogeneous_quorum_tail([0.5], 1.5)  # type: ignore[arg-type]


def test_conservative_filter_removes_largest_probabilities() -> None:
    probabilities = [0.1, 0.9, 0.7, 0.2]

    assert remove_largest_probabilities(probabilities, 0) == tuple(probabilities)
    assert remove_largest_probabilities(probabilities, 2) == (0.1, 0.2)
    assert remove_largest_probabilities(probabilities, 10) == ()


def test_conservative_quorum_tail_is_no_larger_than_unfiltered_tail() -> None:
    probabilities = [0.2, 0.5, 0.8, 0.9]

    unfiltered = heterogeneous_quorum_tail(probabilities, 2)
    filtered = conservative_quorum_tail(probabilities, quorum_size=2, fault_count=1)

    assert filtered <= unfiltered


def test_quorum_tail_result_is_diagnostic_not_metric() -> None:
    result = evaluate_quorum_tail([0.2, 0.5, 0.8], quorum_size=2)

    assert result.evaluator_id == QUORUM_TAIL_EVALUATOR_ID
    assert result.quorum_size == 2
    assert result.input_count == 3
    assert 0.0 <= result.probability <= 1.0


def test_quorum_tail_source_avoids_sampling_enumeration_and_v5_imports() -> None:
    source = (ROOT / "src" / "marl_topology" / "protocol" / "quorum_tail.py").read_text(
        encoding="utf-8"
    )

    banned_terms = [
        "itertools",
        "combinations",
        "permutations",
        "monte_carlo",
        "Monte Carlo",
        "random",
        "sample",
        "import v5",
        "from v5",
        "torch",
    ]
    hits = [term for term in banned_terms if term in source]

    assert not hits, f"quorum tail utility uses forbidden implementation routes: {hits}"
