from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_doc(name: str) -> str:
    return _read(ROOT / "docs" / name)


def test_stage5_0e_fixture_family_design_exists_and_is_design_only() -> None:
    text = _read_doc("STAGE5_0E_TAU_CONSENSUS_FIXTURE_FAMILY_DESIGN.md")

    required = [
        "Stage 5.0e",
        "fixture-family design only",
        "does not implement fixtures",
        "does not run calibration",
        "does not select",
        "`tau_consensus`",
        "does not implement reward",
        "does not train models",
        "does not migrate v5 code",
        "Stage 4.8 remains a smoke-test seed",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 5.0e design missing boundary terms: {missing}"


def test_stage5_0e_design_lists_required_fixture_families_and_axes() -> None:
    text = _read_doc("STAGE5_0E_TAU_CONSENSUS_FIXTURE_FAMILY_DESIGN.md")

    required_families = [
        "clear_free_space_reference",
        "near_threshold_link_budget",
        "blocked_or_nlos_urban",
        "urban_gap_or_corridor_los",
        "rsu_height_recovery",
        "same_resource_interference",
        "orthogonal_resource_mitigation",
        "deadline_tight_retransmission",
        "unreachable_reliability_target",
        "sparse_vs_dense_tradeoff",
        "multi_hop_route",
        "resource_redundant_topology",
        "failed_scheduled_message",
        "weak_primary_distribution",
        "center_vs_edge_primary",
        "symmetric_pbft_reference",
    ]
    missing_families = [family for family in required_families if family not in text]
    assert not missing_families, f"missing fixture families: {missing_families}"

    required_axes = [
        "geometry",
        "channel/link",
        "network resource",
        "topology trade-off",
        "PBFT primary asymmetry",
        "deadline",
    ]
    missing_axes = [axis for axis in required_axes if axis not in text]
    assert not missing_axes, f"missing coverage axes: {missing_axes}"


def test_stage5_0e_design_declares_manifest_fields_and_metric_governance() -> None:
    text = _read_doc("STAGE5_0E_TAU_CONSENSUS_FIXTURE_FAMILY_DESIGN.md")

    required = [
        "`scenario_set_id`",
        "`family_id`",
        "`coverage_axis`",
        "`geometry_visibility_regime`",
        "`link_transmission_regime`",
        "`pbft_model_id`",
        "`committee_size`",
        "`fault_tolerance`",
        "`topology_variant_ids`",
        "`consensus_success_probability`",
        "`latency`",
        "`energy`",
        "`topology_diagnostics`",
        "Stage 5.0e adds no metric names",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"missing manifest or metric-governance terms: {missing}"


def test_stage5_0e_design_protects_oracle_and_full_graph_boundaries() -> None:
    text = _read_doc("STAGE5_0E_TAU_CONSENSUS_FIXTURE_FAMILY_DESIGN.md")

    required = [
        "full graph remains a baseline, not an oracle",
        "Full graph remains baseline, not oracle",
        "oracle candidates are review-only diagnostics",
        "must not enter deployment actor observations",
        "density is not a primary objective",
        "redundant edges do not get reliability bonus above tau",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"missing oracle/full-graph boundary terms: {missing}"


def test_stage5_0e_updates_stage5_contracts_and_plan() -> None:
    docs = {
        "TAU_CONSENSUS_CALIBRATION_PLAN": _read_doc("TAU_CONSENSUS_CALIBRATION_PLAN.md"),
        "OBJECTIVE_CONTRACT": _read_doc("OBJECTIVE_CONTRACT.md"),
        "REWARD_CONTRACT": _read_doc("REWARD_CONTRACT.md"),
        "REWARD_SURROGATE_CONTRACT": _read_doc("REWARD_SURROGATE_CONTRACT.md"),
        "METRIC_CONTRACT": _read_doc("METRIC_CONTRACT.md"),
        "STAGE5_REWARD_OBJECTIVE_FREEZE": _read_doc("STAGE5_REWARD_OBJECTIVE_FREEZE.md"),
        "STAGE5_0D": _read_doc("STAGE5_0D_TAU_CONSENSUS_CALIBRATION_REPORT.md"),
    }

    for name, text in docs.items():
        assert "Stage 5.0e" in text, f"{name} missing Stage 5.0e reference"
    assert "STAGE5_0E_TAU_CONSENSUS_FIXTURE_FAMILY_DESIGN.md" in docs["TAU_CONSENSUS_CALIBRATION_PLAN"]


def test_stage5_0e_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage5_0e_tau_consensus_fixture_family_design.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage5_0e_tau_consensus_fixture_family_design"
    for field in [
        "required_artifacts",
        "expected_evidence",
        "negative_checks",
        "relevant_lessons",
        "required_tests",
        "forbidden_v5_inheritance",
        "expected_outputs",
    ]:
        assert field in task
        assert task[field]
    negative_text = " ".join(task["negative_checks"])
    assert "final tau_consensus" in negative_text
    assert "fixtures are implemented" in negative_text
    assert "calibration is run" in negative_text
    assert "full graph" in negative_text
    assert "v5" in negative_text


def test_stage5_0e_project_state_marks_completion_and_blocks_final_tau() -> None:
    text = _read_doc("PROJECT_STATE.md")

    required = [
        "post_stage_5_0e_awaiting_owner_decision",
        "post_stage_5_0d_awaiting_owner_decision",
        "stage_5_0e_tau_consensus_fixture_family_design",
        "stage5_0e_tau_consensus_fixture_family_design_gate",
        "tau_consensus_final_selection",
        "reward_implementation",
        "reward_weight_calibration",
        "training_runs",
        "v5_code_migration",
        "recommended_next_task: stage_5_0f_tau_consensus_fixture_implementation_plan_without_run",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 5.0e state: {missing}"


def test_stage5_0e_does_not_add_forbidden_reward_training_model_or_v5_code() -> None:
    banned_terms = [
        "class Actor(",
        "class Critic(",
        "COMA",
        "MAPPO",
        "optimizer",
        "train_loop",
        "def reward",
        "def compute_reward",
        "class Reward",
        "reward =",
        "reward:",
        "P_eff_soft",
        "P_eff_hard",
        "hard_eval",
        "soft_train",
        "D:\\PhD_works\\v5",
    ]
    offenders: dict[str, list[str]] = {}
    for path in (ROOT / "src" / "marl_topology").rglob("*.py"):
        text = _read(path)
        hits = [term for term in banned_terms if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"Stage 5.0e added forbidden source terms: {offenders}"
