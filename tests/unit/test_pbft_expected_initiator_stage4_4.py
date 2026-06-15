from pathlib import Path

import pytest

from marl_topology.protocol import (
    FAULT_FILTER_NONE,
    FAULT_FILTER_REMOVE_LARGEST,
    PBFT_EXPECTED_INITIATOR_MODEL_ID,
    PBFTExpectedInitiatorConfig,
    PBFTThreePhaseConfig,
    evaluate_expected_initiator_pbft_reliability,
    evaluate_pbft_given_primary,
    evaluate_pbft_three_phase_reliability,
)


ROOT = Path(__file__).resolve().parents[2]
NODES = ("n0", "n1", "n2", "n3")


def _complete_matrix(probability: float) -> dict[tuple[str, str], float]:
    return {
        (sender_id, receiver_id): probability
        for sender_id in NODES
        for receiver_id in NODES
        if sender_id != receiver_id
    }


def _config(
    *,
    fault_filter_mode: str = FAULT_FILTER_NONE,
) -> PBFTExpectedInitiatorConfig:
    return PBFTExpectedInitiatorConfig(
        node_ids=NODES,
        fault_tolerance=1,
        fault_filter_mode=fault_filter_mode,
    )


def test_symmetric_complete_graph_has_equal_primary_reliability() -> None:
    matrix = _complete_matrix(0.82)
    record = evaluate_expected_initiator_pbft_reliability(
        _config(),
        pre_prepare_matrix=matrix,
        prepare_matrix=matrix,
        commit_matrix=matrix,
    )

    values = tuple(record.per_primary_reliability.values())
    assert record.model_id == PBFT_EXPECTED_INITIATOR_MODEL_ID
    assert len(set(round(value, 12) for value in values)) == 1
    assert record.consensus_success_probability == pytest.approx(values[0])
    assert record.primary_distribution == {node_id: 0.25 for node_id in NODES}
    assert record.mean_field_assumption is True
    assert record.view_change_mode == "deferred"


def test_weak_primary_has_lower_primary_specific_reliability_but_average_is_topology_metric() -> None:
    pre_prepare = _complete_matrix(0.92)
    prepare = _complete_matrix(0.92)
    commit = _complete_matrix(0.92)
    for receiver_id in NODES:
        if receiver_id != "n3":
            pre_prepare[("n3", receiver_id)] = 0.15

    record = evaluate_expected_initiator_pbft_reliability(
        _config(),
        pre_prepare_matrix=pre_prepare,
        prepare_matrix=prepare,
        commit_matrix=commit,
    )
    manual_average = sum(record.per_primary_reliability.values()) / len(NODES)

    assert record.per_primary_reliability["n3"] < record.per_primary_reliability["n0"]
    assert record.consensus_success_probability == pytest.approx(manual_average)


def test_improving_one_primary_outgoing_links_improves_that_primary_and_average() -> None:
    weak_pre_prepare = _complete_matrix(0.85)
    prepare = _complete_matrix(0.85)
    commit = _complete_matrix(0.85)
    for receiver_id in NODES:
        if receiver_id != "n2":
            weak_pre_prepare[("n2", receiver_id)] = 0.2

    improved_pre_prepare = dict(weak_pre_prepare)
    for receiver_id in NODES:
        if receiver_id != "n2":
            improved_pre_prepare[("n2", receiver_id)] = 0.95

    baseline = evaluate_expected_initiator_pbft_reliability(
        _config(),
        pre_prepare_matrix=weak_pre_prepare,
        prepare_matrix=prepare,
        commit_matrix=commit,
    )
    improved = evaluate_expected_initiator_pbft_reliability(
        _config(),
        pre_prepare_matrix=improved_pre_prepare,
        prepare_matrix=prepare,
        commit_matrix=commit,
    )

    assert improved.per_primary_reliability["n2"] > baseline.per_primary_reliability["n2"]
    assert improved.consensus_success_probability > baseline.consensus_success_probability


def test_primary_specific_helper_matches_fixed_primary_path() -> None:
    matrix = _complete_matrix(0.77)
    expected_config = _config()
    helper = evaluate_pbft_given_primary(
        "n1",
        expected_config,
        pre_prepare_matrix=matrix,
        prepare_matrix=matrix,
        commit_matrix=matrix,
    )
    fixed_primary = evaluate_pbft_three_phase_reliability(
        PBFTThreePhaseConfig(
            node_ids=NODES,
            primary_id="n1",
            fault_tolerance=1,
            fault_filter_mode=FAULT_FILTER_NONE,
        ),
        pre_prepare_matrix=matrix,
        prepare_matrix=matrix,
        commit_matrix=matrix,
    )

    assert helper.primary_id == "n1"
    assert helper.consensus_success_probability == pytest.approx(
        fixed_primary.consensus_success_probability
    )


def test_remove_largest_fault_filter_lowers_or_preserves_expected_reliability() -> None:
    matrix = _complete_matrix(0.7)
    no_filter = evaluate_expected_initiator_pbft_reliability(
        _config(fault_filter_mode=FAULT_FILTER_NONE),
        pre_prepare_matrix=matrix,
        prepare_matrix=matrix,
        commit_matrix=matrix,
    )
    filtered = evaluate_expected_initiator_pbft_reliability(
        _config(fault_filter_mode=FAULT_FILTER_REMOVE_LARGEST),
        pre_prepare_matrix=matrix,
        prepare_matrix=matrix,
        commit_matrix=matrix,
    )

    assert filtered.consensus_success_probability <= no_filter.consensus_success_probability


def test_expected_initiator_config_rejects_unsupported_modes() -> None:
    with pytest.raises(ValueError, match="uniform"):
        PBFTExpectedInitiatorConfig(
            node_ids=NODES,
            fault_tolerance=1,
            primary_distribution="fixed",
        )
    with pytest.raises(ValueError, match="self_vote_counted"):
        PBFTExpectedInitiatorConfig(
            node_ids=NODES,
            fault_tolerance=1,
            self_vote_counted=False,
        )


def test_stage4_4_source_avoids_random_sampling_and_subset_enumeration_routes() -> None:
    source = (ROOT / "src" / "marl_topology" / "protocol" / "pbft_reliability.py").read_text(
        encoding="utf-8"
    )

    banned_terms = [
        "itertools",
        "combinations",
        "permutations",
        "monte_carlo",
        "Monte Carlo",
        "random",
        "import v5",
        "from v5",
        "torch",
        "class Actor(",
        "class Critic(",
        "COMA",
        "MAPPO",
        "optimizer",
        "backward(",
        "train_loop",
        "torch.save",
        "P_eff",
    ]
    hits = [term for term in banned_terms if term in source]

    assert not hits, f"PBFT expected-initiator source uses forbidden routes: {hits}"
