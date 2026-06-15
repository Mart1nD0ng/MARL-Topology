from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_stage19_document_records_order_results_and_boundaries() -> None:
    text = _read("docs/STAGE19_SUPERVISED_ACTOR_STACK_RERUN.md")

    required = [
        "MLP -> GNN -> GRU real multi-step subset -> LSTM after GRU sanity",
        "stage18_disambiguated_actor_evidence_v1",
        "actor_edge_utility_target",
        "pairwise ranking",
        "GNN validation loss was lower than MLP",
        "LSTM matched or slightly improved GRU",
        "stage_20_supervised_actor_policy_evaluation_with_environment_assembler",
        "No checkpoint was written",
        "No training artifact was written",
    ]
    missing = [term for term in required if term not in text]
    assert not missing, f"Stage 19 doc missing terms: {missing}"


def test_stage19_harness_task_exists_and_registers_closeout_gate() -> None:
    task = yaml.safe_load(_read("harness/tasks/stage19_supervised_actor_stack_rerun.yaml"))

    assert task["id"] == "stage19_supervised_actor_stack_rerun"
    expected = " ".join(task["expected_evidence"])
    assert "MLP supervised rerun" in expected
    assert "GNN supervised rerun" in expected
    assert "GRU is tested on a real multi-step" in expected
    assert "LSTM is tested only after GRU sanity passes" in expected
    negative = " ".join(task["negative_checks"])
    assert "PPO MAPPO" in negative
    assert "COMA" in negative
    assert "Transformer" in negative
    assert "checkpoint" in negative


def test_stage19_project_state_marks_stage19_complete_and_stage20_owner_gate() -> None:
    text = _read("docs/PROJECT_STATE.md")

    required = [
        "current_stage: post_stage_19_complete_awaiting_owner_decision_for_stage_20",
        "stage_19_rerun_supervised_actor_stack_on_disambiguated_evidence",
        "recommended_next_task: stage_20_supervised_actor_policy_evaluation_with_environment_assembler",
        "stage19_supervised_actor_stack_rerun_gate",
        "stage19_no_policy_gradient_checkpoint_artifact_gate",
        "PPO_MAPPO_rerun_until_stage20_owner_approval",
        "COMA",
        "Transformer",
        "owner_decision_required: true",
    ]
    missing = [term for term in required if term not in text]
    assert not missing, f"PROJECT_STATE missing Stage 19 state: {missing}"


def test_stage19_script_prints_report_and_writes_no_artifact() -> None:
    text = _read("scripts/train/stage19_supervised_actor_stack.py")

    required = [
        "run_stage19_supervised_actor_stack",
        "print(json.dumps",
        "Stage 19 supervised actor stack rerun",
    ]
    missing = [term for term in required if term not in text]
    assert not missing

    forbidden = [
        "torch.save",
        "checkpoint_path",
        "write_text(",
        "write_bytes(",
        "mkdir(",
        "D:\\PhD_works\\v5",
    ]
    hits = [term for term in forbidden if term in text]
    assert not hits, f"Stage 19 script introduced forbidden writes: {hits}"


def test_stage19_source_does_not_add_forbidden_rl_or_global_actor_paths() -> None:
    paths = [
        "src/marl_topology/training/stage19_supervised_actor_stack.py",
        "scripts/train/stage19_supervised_actor_stack.py",
    ]
    text = "\n".join(_read(path) for path in paths)

    forbidden = [
        "PPOTrainer",
        "MAPPOTrainer",
        "class COMA",
        "class Transformer",
        "torch.save",
        "checkpoint_path",
        "reward_weight_tuning = True",
        "final_tau_selected = True",
        "D:\\PhD_works\\v5",
    ]
    hits = [term for term in forbidden if term in text]
    assert not hits, f"Stage 19 source introduced forbidden paths: {hits}"
