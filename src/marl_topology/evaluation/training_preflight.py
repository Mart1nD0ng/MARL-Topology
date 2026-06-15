"""Stage 5.5 training preflight review without execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .surrogate_diagnostics_report import build_stage5_4_reward_report_integration


STAGE5_5_TRAINING_PREFLIGHT_STAGE_ID = (
    "stage_5_5_training_preflight_review_without_training"
)
STAGE5_5_RECOMMENDED_NEXT_TASK = (
    "stage_5_6_training_design_contract_without_execution"
)
STAGE5_5_VERDICT = "not_ready_for_training_execution"


@dataclass(frozen=True, slots=True)
class TrainingPreflightGate:
    """One gate in the Stage 5.5 preflight review."""

    gate_id: str
    status: str
    controlled_object: str
    evidence: str
    required_next_action: str
    blocks_training_execution: bool

    def __post_init__(self) -> None:
        if self.status not in {"pass", "blocked", "deferred"}:
            raise ValueError("status must be pass, blocked, or deferred")
        for field_name in (
            "gate_id",
            "controlled_object",
            "evidence",
            "required_next_action",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if self.status == "blocked" and not self.blocks_training_execution:
            raise ValueError("blocked gates must block training execution")

    def to_payload(self) -> dict[str, object]:
        return {
            "gate_id": self.gate_id,
            "status": self.status,
            "controlled_object": self.controlled_object,
            "evidence": self.evidence,
            "required_next_action": self.required_next_action,
            "blocks_training_execution": self.blocks_training_execution,
        }


def build_stage5_5_training_preflight_review() -> dict[str, object]:
    """Build the Stage 5.5 training preflight review report."""

    stage5_4_report = build_stage5_4_reward_report_integration()
    gates = _preflight_gates(stage5_4_report)
    payload_gates = [gate.to_payload() for gate in gates]
    pass_count = sum(1 for gate in gates if gate.status == "pass")
    blocked_count = sum(1 for gate in gates if gate.status == "blocked")
    deferred_count = sum(1 for gate in gates if gate.status == "deferred")
    blocking_gate_ids = tuple(gate.gate_id for gate in gates if gate.status == "blocked")

    return {
        "stage": STAGE5_5_TRAINING_PREFLIGHT_STAGE_ID,
        "source_stage": str(stage5_4_report["stage"]),
        "verdict": STAGE5_5_VERDICT,
        "training_execution_allowed": False,
        "training_design_contract_allowed": True,
        "recommended_next_task": STAGE5_5_RECOMMENDED_NEXT_TASK,
        "owner_decision_required": True,
        "gate_summary": {
            "pass_count": pass_count,
            "blocked_count": blocked_count,
            "deferred_count": deferred_count,
            "blocking_gate_ids": list(blocking_gate_ids),
        },
        "gates": payload_gates,
        "required_before_training_execution": list(
            _required_before_training_execution()
        ),
        "allowed_next_tasks": [
            STAGE5_5_RECOMMENDED_NEXT_TASK,
            "stage_5_5_training_preflight_review_refinement",
        ],
        "blocked_tasks": [
            "training_execution",
            "reward_weight_calibration",
            "actor_critic_model_implementation",
            "final_tau_selection_without_owner_decision",
            "legacy_code_migration",
        ],
        "checks": _checks(stage5_4_report, gates),
    }


def _preflight_gates(
    stage5_4_report: Mapping[str, object],
) -> tuple[TrainingPreflightGate, ...]:
    checks = stage5_4_report["checks"]
    if not isinstance(checks, Mapping):
        raise ValueError("Stage 5.4 checks must be a mapping")

    return (
        TrainingPreflightGate(
            gate_id="metric_governance_ready",
            status="pass",
            controlled_object="registered evaluation quantities",
            evidence="Stage 5.4 report reads only registered consensus, latency, energy, and topology diagnostics concepts.",
            required_next_action="Keep any new training diagnostics outside evaluation metric tables.",
            blocks_training_execution=False,
        ),
        TrainingPreflightGate(
            gate_id="objective_surrogate_boundary_ready",
            status="pass",
            controlled_object="objective and training-side surrogate boundary",
            evidence="Stage 5.4 reports component-only diagnostics and scalar_surrogate_reported is false.",
            required_next_action="Preserve registered objective reporting alongside any future training signal.",
            blocks_training_execution=False,
        ),
        TrainingPreflightGate(
            gate_id="normalization_reference_available",
            status="pass" if checks["normalization_reference_applied"] else "blocked",
            controlled_object="latency and energy normalization references",
            evidence="Stage 5.3 fixed references are applied to every Stage 5.4 diagnostic row.",
            required_next_action="Use fixed references as config inputs only; do not treat them as weights.",
            blocks_training_execution=not bool(checks["normalization_reference_applied"]),
        ),
        TrainingPreflightGate(
            gate_id="dec_pomdp_boundary_ready_for_design",
            status="pass",
            controlled_object="deployment actor information boundary",
            evidence="ActorObservation, replay column, and surrogate diagnostic boundaries keep metrics and diagnostics out of actor inputs.",
            required_next_action="Before architecture work, require an actor-feature contract and leakage-negative tests.",
            blocks_training_execution=False,
        ),
        TrainingPreflightGate(
            gate_id="baseline_oracle_evidence_available",
            status="pass",
            controlled_object="non-learning baseline and oracle evidence",
            evidence="Stage 4 and Stage 5 reports include sparse, dense, failure, and requirement-feasibility evidence.",
            required_next_action="Carry baselines into future training evaluation as comparison sensors.",
            blocks_training_execution=False,
        ),
        TrainingPreflightGate(
            gate_id="component_report_available",
            status="pass",
            controlled_object="pre-training diagnostic observability",
            evidence="Stage 5.4 exposes reliability violation, plateau, normalized latency, and normalized energy components.",
            required_next_action="Use component diagnostics to debug training later, not as final success evidence.",
            blocks_training_execution=False,
        ),
        TrainingPreflightGate(
            gate_id="surrogate_weight_policy_missing",
            status="blocked",
            controlled_object="future scalarization and component weights",
            evidence="Stage 5.4 intentionally performs no weight calibration and reports no scalar surrogate.",
            required_next_action="Define a Stage 5.6 design contract before any scalarization or weight choice.",
            blocks_training_execution=True,
        ),
        TrainingPreflightGate(
            gate_id="return_advantage_target_contract_missing",
            status="blocked",
            controlled_object="future learning targets and replay columns",
            evidence="Replay contract still admits only surrogate diagnostics; return, advantage, and value-target fields remain unsupported.",
            required_next_action="Create a learning-target and replay-column contract before learner implementation.",
            blocks_training_execution=True,
        ),
        TrainingPreflightGate(
            gate_id="actor_critic_architecture_contract_missing",
            status="blocked",
            controlled_object="policy and critic architecture boundary",
            evidence="No actor, critic, memory, graph, or centralized-training architecture contract is frozen.",
            required_next_action="Define architecture options and information boundaries without implementation first.",
            blocks_training_execution=True,
        ),
        TrainingPreflightGate(
            gate_id="training_artifact_policy_missing",
            status="blocked",
            controlled_object="future experiment artifacts",
            evidence="No contract exists for seeds, checkpoints, logs, replay snapshots, or result directories.",
            required_next_action="Define artifact, seed, and run-manifest policy before execution.",
            blocks_training_execution=True,
        ),
        TrainingPreflightGate(
            gate_id="multi_seed_protocol_missing",
            status="blocked",
            controlled_object="stochastic evidence protocol",
            evidence="The project has deterministic alpha fixtures but no repeated-seed protocol for learning claims.",
            required_next_action="Define seeds, variance reporting, stop conditions, and baseline regression checks.",
            blocks_training_execution=True,
        ),
        TrainingPreflightGate(
            gate_id="scenario_distribution_calibration_deferred",
            status="deferred",
            controlled_object="deployment scenario distribution",
            evidence="Stage 5 evidence uses deterministic alpha fixtures and Stage 3-backed sweeps, not a deployment-calibrated city distribution.",
            required_next_action="Calibrate or declare scenario distributions before claiming deployment generalization.",
            blocks_training_execution=False,
        ),
        TrainingPreflightGate(
            gate_id="credit_calibration_deferred",
            status="deferred",
            controlled_object="multi-agent credit assignment",
            evidence="No centralized critic or edge-credit calibration is selected.",
            required_next_action="Review credit assignment only after actor-local schema and learning targets are contracted.",
            blocks_training_execution=False,
        ),
        TrainingPreflightGate(
            gate_id="tau_final_selection_deferred",
            status="deferred",
            controlled_object="formal objective threshold",
            evidence="Owner requirement baseline is 0.9, but final tau selection remains an owner decision.",
            required_next_action="Keep tau requirement and final tau decision separate in future configs.",
            blocks_training_execution=False,
        ),
    )


def _required_before_training_execution() -> tuple[str, ...]:
    return (
        "stage_5_6_training_design_contract_without_execution",
        "surrogate_scalarization_and_weight_policy",
        "learning_target_and_replay_column_contract",
        "actor_critic_architecture_contract",
        "artifact_seed_and_run_manifest_policy",
        "multi_seed_evaluation_protocol",
        "owner_approval_for_training_execution",
    )


def _checks(
    stage5_4_report: Mapping[str, object],
    gates: tuple[TrainingPreflightGate, ...],
) -> dict[str, object]:
    stage5_4_checks = stage5_4_report["checks"]
    if not isinstance(stage5_4_checks, Mapping):
        raise ValueError("Stage 5.4 checks must be a mapping")

    return {
        "source_stage5_4_report_available": (
            stage5_4_report["stage"]
            == "stage_5_4_reward_report_integration_without_training"
        ),
        "component_only_report_policy": (
            stage5_4_report["component_policy"]
            == "component_only_no_scalar_reward_v1"
        ),
        "scalar_surrogate_not_reported": bool(
            stage5_4_checks["scalar_surrogate_not_reported"]
        ),
        "normalization_reference_applied": bool(
            stage5_4_checks["normalization_reference_applied"]
        ),
        "surrogate_outputs_not_metrics": bool(
            stage5_4_checks["surrogate_outputs_not_metrics"]
        ),
        "training_execution_allowed": False,
        "training_run": False,
        "model_code_added": False,
        "v5_code_migrated": False,
        "reward_weight_calibration_performed": False,
        "final_tau_selected": False,
        "all_blocking_gates_visible": any(
            gate.status == "blocked" and gate.blocks_training_execution
            for gate in gates
        ),
        "recommended_next_task": STAGE5_5_RECOMMENDED_NEXT_TASK,
    }
