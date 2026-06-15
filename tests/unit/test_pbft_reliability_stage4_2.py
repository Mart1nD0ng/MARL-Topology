from pathlib import Path

import pytest

from marl_topology.protocol import (
    FAULT_FILTER_NONE,
    FAULT_FILTER_REMOVE_LARGEST,
    PBFT_PHASE_NAMES,
    PBFT_RELIABILITY_VARIANT_ID,
    PBFTThreePhaseConfig,
    evaluate_pbft_three_phase_reliability,
)


ROOT = Path(__file__).resolve().parents[2]
NODES = ("n0", "n1", "n2", "n3")


def _complete_matrix(probability: float, node_ids: tuple[str, ...] = NODES) -> dict[tuple[str, str], float]:
    return {
        (sender_id, receiver_id): probability
        for sender_id in node_ids
        for receiver_id in node_ids
        if sender_id != receiver_id
    }


def _config(
    *,
    fault_filter_mode: str = FAULT_FILTER_REMOVE_LARGEST,
) -> PBFTThreePhaseConfig:
    return PBFTThreePhaseConfig(
        node_ids=NODES,
        primary_id="n0",
        fault_tolerance=1,
        fault_filter_mode=fault_filter_mode,
    )


def _evaluate(
    probability: float,
    *,
    fault_filter_mode: str = FAULT_FILTER_REMOVE_LARGEST,
):
    matrix = _complete_matrix(probability)
    return evaluate_pbft_three_phase_reliability(
        _config(fault_filter_mode=fault_filter_mode),
        pre_prepare_matrix=matrix,
        prepare_matrix=matrix,
        commit_matrix=matrix,
    )


def test_pbft_config_rejects_invalid_committee_and_fault_parameters() -> None:
    with pytest.raises(ValueError, match="n >= 3f"):
        PBFTThreePhaseConfig(
            node_ids=("n0", "n1", "n2"),
            primary_id="n0",
            fault_tolerance=1,
        )
    with pytest.raises(ValueError, match="primary_id"):
        PBFTThreePhaseConfig(
            node_ids=NODES,
            primary_id="missing",
            fault_tolerance=1,
        )
    with pytest.raises(ValueError, match="fault_tolerance"):
        PBFTThreePhaseConfig(
            node_ids=NODES,
            primary_id="n0",
            fault_tolerance=-1,
        )


def test_all_one_declared_matrices_produce_reliable_consensus() -> None:
    record = _evaluate(1.0)

    assert record.protocol_variant == PBFT_RELIABILITY_VARIANT_ID
    assert record.phase_names == PBFT_PHASE_NAMES
    assert record.total_quorum == 3
    assert record.external_quorum == 2
    assert record.consensus_success_probability == pytest.approx(1.0)
    assert set(record.pre_prepare_readiness) == set(NODES)
    assert set(record.prepared_probability) == set(NODES)
    assert set(record.committed_probability) == set(NODES)
    assert record.mean_field_assumption is True
    assert record.uses_declared_matrices_only is True
    assert record.uses_stage3_adapter is False


def test_zero_prepare_or_commit_phase_prevents_consensus() -> None:
    ones = _complete_matrix(1.0)
    zeros = _complete_matrix(0.0)
    config = _config()

    zero_prepare = evaluate_pbft_three_phase_reliability(
        config,
        pre_prepare_matrix=ones,
        prepare_matrix=zeros,
        commit_matrix=ones,
    )
    zero_commit = evaluate_pbft_three_phase_reliability(
        config,
        pre_prepare_matrix=ones,
        prepare_matrix=ones,
        commit_matrix=zeros,
    )

    assert zero_prepare.consensus_success_probability == 0.0
    assert zero_commit.consensus_success_probability == 0.0


def test_reliability_is_monotonic_when_a_message_probability_improves() -> None:
    config = _config(fault_filter_mode=FAULT_FILTER_NONE)
    pre_prepare = _complete_matrix(0.65)
    prepare = _complete_matrix(0.65)
    commit = _complete_matrix(0.65)

    baseline = evaluate_pbft_three_phase_reliability(
        config,
        pre_prepare_matrix=pre_prepare,
        prepare_matrix=prepare,
        commit_matrix=commit,
    )
    improved_prepare = dict(prepare)
    improved_prepare[("n1", "n2")] = 0.95
    improved = evaluate_pbft_three_phase_reliability(
        config,
        pre_prepare_matrix=pre_prepare,
        prepare_matrix=improved_prepare,
        commit_matrix=commit,
    )

    assert improved.consensus_success_probability >= baseline.consensus_success_probability


def test_conservative_fault_filter_is_no_larger_than_no_filter() -> None:
    matrix = _complete_matrix(0.7)
    no_filter = evaluate_pbft_three_phase_reliability(
        _config(fault_filter_mode=FAULT_FILTER_NONE),
        pre_prepare_matrix=matrix,
        prepare_matrix=matrix,
        commit_matrix=matrix,
    )
    filtered = evaluate_pbft_three_phase_reliability(
        _config(fault_filter_mode=FAULT_FILTER_REMOVE_LARGEST),
        pre_prepare_matrix=matrix,
        prepare_matrix=matrix,
        commit_matrix=matrix,
    )

    assert filtered.consensus_success_probability <= no_filter.consensus_success_probability


def test_sparse_or_missing_declared_matrices_are_treated_as_zero_delivery() -> None:
    config = _config()
    sparse_pre_prepare = {("n0", "n1"): 1.0}
    ones = _complete_matrix(1.0)

    record = evaluate_pbft_three_phase_reliability(
        config,
        pre_prepare_matrix=sparse_pre_prepare,
        prepare_matrix=ones,
        commit_matrix=ones,
    )

    assert record.pre_prepare_readiness["n0"] == 1.0
    assert record.pre_prepare_readiness["n1"] == 1.0
    assert record.pre_prepare_readiness["n2"] == 0.0
    assert record.pre_prepare_readiness["n3"] == 0.0
    assert record.consensus_success_probability == 0.0


def test_message_matrices_reject_invalid_entries() -> None:
    config = _config()
    ones = _complete_matrix(1.0)

    with pytest.raises(ValueError, match="self messages"):
        evaluate_pbft_three_phase_reliability(
            config,
            pre_prepare_matrix={("n0", "n0"): 1.0},
            prepare_matrix=ones,
            commit_matrix=ones,
        )
    with pytest.raises(ValueError, match="endpoints"):
        evaluate_pbft_three_phase_reliability(
            config,
            pre_prepare_matrix={("n0", "missing"): 1.0},
            prepare_matrix=ones,
            commit_matrix=ones,
        )
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        evaluate_pbft_three_phase_reliability(
            config,
            pre_prepare_matrix={("n0", "n1"): 1.2},
            prepare_matrix=ones,
            commit_matrix=ones,
        )


def test_stage4_2_source_avoids_sampling_training_and_v5_imports() -> None:
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
        "sample",
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

    assert not hits, f"PBFT reliability source uses forbidden routes: {hits}"
