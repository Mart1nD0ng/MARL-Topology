import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage8_0_document_exists_and_states_interface_not_model() -> None:
    text = _read_doc("STAGE8_0_ACTOR_POLICY_INTERFACE_CONTRACT.md")

    required = [
        "Stage 8.0 Actor Policy Interface Contract With Owner Approval",
        "actor_policy_interface_contract_frozen_implementation_blocked",
        "implementation_allowed = false",
        "training_execution_allowed = false",
        "actor_policy_local_input_v1",
        "actor_policy_local_edge_score_output_v1",
        "actor_policy_local_edge_decision_v1",
        "`activate` belongs only to the legacy local decision schema",
        "objective metrics",
        "Joint topology assembly",
        "does not implement actor, critic, graph, recurrent, checkpoint, or training code",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 8.0 doc missing terms: {missing}"


def test_stage8_0_updates_contract_docs_and_project_state() -> None:
    policy = _read_doc("POLICY_ARCHITECTURE_CONTRACT.md")
    training = _read_doc("TRAINING_CONTRACT.md")
    metric = _read_doc("METRIC_CONTRACT.md")
    dec_pomdp = _read_doc("DEC_POMDP_CONTRACT.md")
    state = _read_doc("PROJECT_STATE.md")

    assert "Stage 8.0 Actor Policy Interface" in policy
    assert "Stage 8.0 Actor Policy Interface Boundary" in training
    assert "Stage 8.0 does not add metric names" in metric
    assert "Stage 8.0 Actor Policy Interface Contract" in dec_pomdp
    for expected in [
        "post_stage_8_0_awaiting_owner_decision_for_stage_8_1",
        "stage_8_0_actor_policy_interface_contract_with_owner_approval",
        "stage8_0_actor_policy_interface_contract_gate",
        "stage_9_0_local_mlp_edge_scorer_baseline_without_training_execution",
        "recommended_next_task: stage_9_0_local_mlp_edge_scorer_baseline_without_training_execution",
        "actor_policy_input_schema_gate",
        "actor_policy_output_schema_gate",
        "owner_decision_required: true",
    ]:
        assert expected in state


def test_stage8_0_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage8_0_actor_policy_interface_contract.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage8_0_actor_policy_interface_contract"
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
    assert "actor input schema includes objective metrics or oracle labels" in negative_text
    assert "actor or critic model implementation is added" in negative_text
    assert "joint topology assembly is moved into the actor interface" in negative_text


def test_stage8_0_replay_script_prints_policy_interface_contract() -> None:
    script = ROOT / "scripts" / "replay" / "stage8_0_actor_policy_interface_contract.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = completed.stdout
    assert '"stage": "stage_8_0_actor_policy_interface_contract_with_owner_approval"' in stdout
    assert '"verdict": "actor_policy_interface_contract_frozen_implementation_blocked"' in stdout
    assert '"implementation_allowed": false' in stdout
    assert '"training_execution_allowed": false' in stdout
    assert '"deployment_actor_boundary_local_only": true' in stdout
    assert '"actor_policy_local_input_v1"' in stdout
    assert '"actor_policy_local_edge_score_output_v1"' in stdout
    assert '"legacy_edge_decision_schema_id": "actor_policy_local_edge_decision_v1"' in stdout
    assert '"recommended_next_task": "stage_8_1_actor_policy_interface_skeleton_without_model_or_training"' in stdout


def test_stage8_0_source_keeps_training_model_checkpoint_v5_and_legacy_metric_out() -> None:
    banned_terms = [
        "def compute_reward",
        "class Reward",
        "reward =",
        "reward:",
        "optimizer.step",
        "train_loop",
        "D:\\PhD_works\\v5",
        "P_eff_soft",
        "P_eff_hard",
        "hard_eval",
        "soft_train",
        "class COMA",
        "class MAPPO",
    ]
    hits: list[str] = []
    for path in (ROOT / "src" / "marl_topology").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for term in banned_terms:
            if term in text:
                hits.append(f"{path.relative_to(ROOT)}:{term}")
    assert not hits, f"Stage 8.0 source introduced forbidden terms: {hits}"
