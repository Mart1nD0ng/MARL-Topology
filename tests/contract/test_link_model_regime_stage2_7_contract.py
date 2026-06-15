from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_link_model_regime_review_doc_exists_and_records_boundaries() -> None:
    text = (ROOT / "docs" / "LINK_MODEL_REGIME_REVIEW.md").read_text(encoding="utf-8")

    required = [
        "Stage 2.7",
        "Link Model Regime Review",
        "stage2_deterministic_distance",
        "link_success_probability = exp",
        "Omitted Components",
        "SINR",
        "interference",
        "not a reward definition",
        "not a full 3D V2X physical simulation",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"LINK_MODEL_REGIME_REVIEW missing terms: {missing}"


def test_physics_contract_records_stage_2_7_regime_boundary() -> None:
    text = (ROOT / "docs" / "PHYSICS_CONTRACT.md").read_text(encoding="utf-8")

    required = [
        "Stage 2.7 Link Regime Review",
        "stage2_deterministic_distance",
        "src/marl_topology/link/regime.py",
        "LoS/NLoS",
        "SINR",
        "interference",
        "new named regime",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PHYSICS_CONTRACT missing Stage 2.7 terms: {missing}"


def test_project_state_marks_stage_2_7_complete_and_waits_for_owner() -> None:
    text = (ROOT / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")

    required = [
        "stage_2_7_link_model_regime_review",
        "link_model_regime_gate",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 2.7 state: {missing}"


def test_link_model_regime_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "link_model_regime_review.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    for field in [
        "relevant_lessons",
        "required_tests",
        "forbidden_v5_inheritance",
        "expected_outputs",
        "negative_checks",
    ]:
        assert field in task
        assert task[field]

    negative_text = " ".join(task["negative_checks"])
    assert "SINR" in negative_text
    assert "reward" in negative_text
    assert "v5 channel" in negative_text


def test_stage_2_7_does_not_add_complex_physics_reward_or_model_code() -> None:
    banned_terms = [
        "class Actor(",
        "class Critic(",
        "COMA",
        "MAPPO",
        "optimizer",
        "train_loop",
        "to_csv",
        "P_eff_soft",
        "P_eff_hard",
        "hard_eval",
        "soft_train",
    ]
    offenders: dict[str, list[str]] = {}
    for path in (ROOT / "src" / "marl_topology").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        hits = [term for term in banned_terms if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"Stage 2.7 added forbidden code or metric defaults: {offenders}"


def test_stage_2_7_does_not_implement_deferred_physics_components() -> None:
    source_paths = []
    for path in (ROOT / "src" / "marl_topology").rglob("*.py"):
        relative_parts = path.relative_to(ROOT).parts
        if path.name == "regime.py":
            continue
        if "channel" in relative_parts:
            continue
        if "link" in relative_parts and path.name in {"transmission.py", "__init__.py"}:
            continue
        if "evaluation" in relative_parts and path.name == "stage3_fixture_suite.py":
            continue
        source_paths.append(path)
    banned_physics_terms = [
        "path_loss_db",
        "shadowing_db",
        "sinr_db",
        "interference_power",
    ]
    offenders: dict[str, list[str]] = {}
    for path in source_paths:
        text = path.read_text(encoding="utf-8")
        hits = [term for term in banned_physics_terms if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"Stage 2.7 implemented deferred physics components: {offenders}"
