from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_ctde_decision_doc_records_decentralized_deployment_boundary() -> None:
    text = (ROOT / "docs" / "CTDE_DEC_POMDP_DECISION.md").read_text(encoding="utf-8")

    required = [
        "Deployment is decentralized",
        "Training may be centralized",
        "CTDE is a boundary",
        "not an immediate commitment",
        "COMA",
        "credit_calibration_gate",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"CTDE decision doc missing: {missing}"


def test_dec_pomdp_contract_names_schema_and_forbidden_fields() -> None:
    text = (ROOT / "docs" / "DEC_POMDP_CONTRACT.md").read_text(encoding="utf-8")

    required = [
        "Stage 2.1 CTDE Decision",
        "Actor Allowed Observations",
        "Actor Forbidden Global Information",
        "CentralizedTrainingView",
        "EdgeActionDecision",
        "JointTopologyAction",
        "global_topology",
        "oracle_label",
        "critic_features",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"DEC_POMDP_CONTRACT missing Stage 2.1 terms: {missing}"


def test_project_state_marks_stage_2_1_completed_and_owner_decision_required() -> None:
    text = (ROOT / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")

    required = [
        "stage_2_1_dec_pomdp_schema_contract",
        "dec_pomdp_leakage_gate",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 2.1 state: {missing}"


def test_no_policy_model_or_training_module_added_for_stage_2_1() -> None:
    source_files = list((ROOT / "src" / "marl_topology").rglob("*.py"))
    banned_terms = [
        "class Actor(",
        "class Critic(",
        "COMA",
        "MAPPO",
        "optimizer",
        "train_loop",
    ]
    offenders: dict[str, list[str]] = {}
    for path in source_files:
        text = path.read_text(encoding="utf-8")
        hits = [term for term in banned_terms if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"Stage 2.1 added forbidden model/training code: {offenders}"
