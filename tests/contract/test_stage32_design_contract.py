"""Stage 32 production-training design contract and decision packet (design-only)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage32_design_and_decision_docs_exist() -> None:
    for name in (
        "STAGE32_PRODUCTION_TRAINING_DESIGN_CONTRACT.md",
        "STAGE32_DECISION_PACKET.md",
    ):
        assert (ROOT / "docs" / name).exists(), name


def test_stage32_design_contract_is_design_only() -> None:
    text = _doc("STAGE32_PRODUCTION_TRAINING_DESIGN_CONTRACT.md")
    assert "Without Execution" in text
    assert "does not authorize training execution" in text.lower() or (
        "design-only" in text.lower()
    )
    # Inherited frozen decisions must be named and unchanged.
    assert "tau_requirement_min = 0.9" in text
    assert "feasibility_first_barrier_v2" in text
    assert "physical_budget_aware_sequential_sampler" in text
    # Named active architectures for the scale-up.
    assert "local_message_passing_gnn_edge_scorer_v2" in text
    assert "centralized_message_passing_graph_value_critic_v1" in text
    # Boundaries kept.
    assert "no v5 migration" in text.lower()
    assert "mean-field" in text.lower()


def test_stage32_decision_packet_requests_owner_decision() -> None:
    text = _doc("STAGE32_DECISION_PACKET.md")
    assert "owner decision required" in text.lower()
    assert "option_b_execute" in text
    assert "scale-up approved: `False`" in text
    # Things that must stay blocked.
    for blocked in ("final tau", "v5 code migration", "COMA"):
        assert blocked in text


def test_stage32_project_state_keeps_execution_owner_gated() -> None:
    state = _doc("PROJECT_STATE.md")
    assert "stage32_production_training_design_contract" in state
    assert (
        "stage32_production_training_execution_without_owner_approval" in state
    )
    assert "owner_decision_required: true" in state
    assert "stage32_execution_owner_gated: true" in state
