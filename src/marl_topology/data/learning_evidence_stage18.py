"""Stage 18 evidence rebuild with disambiguated actor features and targets."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .actor_feature_rebuild import (
    STAGE18_FEATURE_SCHEMA_ID,
    actor_safe_view_has_no_forbidden_fields,
    build_stage18_actor_safe_feature_rows,
    stage18_actor_signature_key,
)
from .actor_label_disambiguation import (
    ContradictionCause,
    analyze_actor_label_disambiguation,
    build_actor_observation_signature,
)
from .disambiguated_targets import (
    TARGET_ROLE_CRITIC_ONLY,
    actor_target_view_has_no_global_delta_fields,
    build_actor_target_views,
    build_critic_target_view,
)
from .learning_evidence import LearningEvidenceDataset, LearningEvidenceRow
from .learning_evidence_stage16 import (
    STAGE16_DATASET_ID,
    STAGE16_SEQUENCE_ID,
    build_stage16_learning_evidence_dataset,
)


STAGE18_STAGE_ID = "stage_18_evidence_rebuild_with_disambiguated_features_and_targets"
STAGE18_DATASET_ID = "stage18_disambiguated_actor_evidence_v1"
STAGE18_VERDICT = "stage18_disambiguated_evidence_ready_for_owner_stage19_decision"
STAGE18_RECOMMENDED_NEXT_TASK_IF_PASS = (
    "stage_19_rerun_supervised_actor_stack_on_disambiguated_evidence"
)
STAGE18_RECOMMENDED_NEXT_TASK_IF_BLOCKED = (
    "stage_18b_actor_feature_boundary_owner_decision_or_target_rebuild"
)
STAGE18_HARD_LABEL_ALLOWED_CONTRADICTION_THRESHOLD = 0.05
STAGE18_MIN_SOFT_UTILITY_COVERAGE = 0.95
STAGE18_MIN_RANKING_PAIR_COUNT = 1


@dataclass(frozen=True, slots=True)
class Stage18EvidenceRow:
    """One rebuilt row with separated Stage 18 views."""

    actor_safe_view: tuple[Mapping[str, object], ...]
    actor_target_view: Mapping[str, object]
    critic_target_view: Mapping[str, object]
    diagnostics_view: Mapping[str, object]

    @property
    def views(self) -> tuple[str, ...]:
        return (
            "actor_safe_view",
            "actor_target_view",
            "critic_target_view",
            "diagnostics_view",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "actor_safe_view": _jsonable(self.actor_safe_view),
            "actor_target_view": _jsonable(self.actor_target_view),
            "critic_target_view": _jsonable(self.critic_target_view),
            "diagnostics_view": _jsonable(self.diagnostics_view),
        }


@dataclass(frozen=True, slots=True)
class Stage18LearningEvidenceDataset:
    """In-memory Stage 18 rebuilt evidence dataset."""

    dataset_id: str
    source_dataset_id: str
    rows: tuple[Stage18EvidenceRow, ...]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def actor_edge_sample_count(self) -> int:
        return sum(len(row.actor_safe_view) for row in self.rows)

    @property
    def soft_target_count(self) -> int:
        return sum(
            len(tuple(row.actor_target_view["actor_soft_utility_targets"]))
            for row in self.rows
        )

    @property
    def ranking_pair_count(self) -> int:
        return sum(
            len(tuple(row.actor_target_view["pairwise_ranking_targets"]))
            for row in self.rows
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "source_dataset_id": self.source_dataset_id,
            "row_count": self.row_count,
            "actor_edge_sample_count": self.actor_edge_sample_count,
            "soft_target_count": self.soft_target_count,
            "ranking_pair_count": self.ranking_pair_count,
            "views": [
                "actor_safe_view",
                "actor_target_view",
                "critic_target_view",
                "diagnostics_view",
            ],
            "rows": [row.to_dict() for row in self.rows],
        }


@dataclass(frozen=True, slots=True)
class Stage18EvidenceBuild:
    dataset: Stage18LearningEvidenceDataset
    report: Mapping[str, object]


def build_stage18_learning_evidence_dataset(
    source_dataset: LearningEvidenceDataset | None = None,
) -> Stage18LearningEvidenceDataset:
    """Rebuild Stage 16 evidence into Stage 18 separated views."""

    source = source_dataset or build_stage16_learning_evidence_dataset()
    before_report = analyze_actor_label_disambiguation(source)
    old_causes = {
        cluster.signature.key: tuple(cluster.suspected_causes)
        for cluster in before_report.contradiction_clusters
    }
    prepared = _prepare_actor_feature_rows(source, old_causes)
    hard_label_allowed_by_signature = _hard_label_allowed_by_signature(prepared)
    rows: list[Stage18EvidenceRow] = []
    for item in prepared:
        source_row = item["source_row"]
        actor_safe_rows = tuple(item["actor_safe_rows"])
        ambiguity_by_signature = dict(item["ambiguity_by_signature"])
        topology_label_role = str(
            source_row.diagnostics.get("topology_label_role", "unknown")
        )
        actor_target_view = build_actor_target_views(
            actor_safe_rows=actor_safe_rows,
            raw_targets=source_row.learning_targets,
            ambiguity_by_signature=ambiguity_by_signature,
            hard_label_allowed_by_signature=hard_label_allowed_by_signature,
            topology_label_role=topology_label_role,
        )
        critic_target_view = build_critic_target_view(
            raw_targets=source_row.learning_targets,
            selected_edges=source_row.selected_edges,
            topology_label_role=topology_label_role,
        )
        diagnostics_view = _diagnostics_view(
            source_row,
            actor_safe_rows=actor_safe_rows,
            actor_target_view=actor_target_view,
            critic_target_view=critic_target_view,
        )
        rows.append(
            Stage18EvidenceRow(
                actor_safe_view=actor_safe_rows,
                actor_target_view=actor_target_view,
                critic_target_view=critic_target_view,
                diagnostics_view=diagnostics_view,
            )
        )
    return Stage18LearningEvidenceDataset(
        dataset_id=STAGE18_DATASET_ID,
        source_dataset_id=source.dataset_id,
        rows=tuple(rows),
    )


def build_stage18_evidence_rebuild_report() -> Stage18EvidenceBuild:
    """Build Stage 18 dataset and before/after diagnostics without writes."""

    source = build_stage16_learning_evidence_dataset()
    before_report = analyze_actor_label_disambiguation(source)
    dataset = build_stage18_learning_evidence_dataset(source)
    after = analyze_stage18_rebuilt_evidence(dataset)
    stage19_allowed = _stage19_allowed(after)
    report = {
        "stage": STAGE18_STAGE_ID,
        "verdict": STAGE18_VERDICT if stage19_allowed else "stage18_rebuild_blocked",
        "dataset_id": dataset.dataset_id,
        "source_dataset_id": source.dataset_id,
        "before": {
            "raw_sample_count": before_report.sample_count,
            "signature_count": before_report.signature_count,
            "contradiction_cluster_count": before_report.contradiction_cluster_count,
            "hard_label_contradiction_rate": before_report.contradiction_rate_by_signature,
        },
        "after": after,
        "stage19_supervised_actor_rerun_allowed": stage19_allowed,
        "recommended_next_task": STAGE18_RECOMMENDED_NEXT_TASK_IF_PASS
        if stage19_allowed
        else STAGE18_RECOMMENDED_NEXT_TASK_IF_BLOCKED,
        "recommended_stage19_model_order": (
            "MLP",
            "GNN",
            "GRU_on_real_multistep_subset",
            "LSTM_after_GRU_sanity",
        )
        if stage19_allowed
        else (),
        "stage19_supervised_actor_stack_rerun_allowed": stage19_allowed,
        "stage11_to_stage15_full_rerun_allowed": False,
        "stage15_policy_gradient_rerun_allowed": False,
        "ppo_mappo_allowed": False,
        "coma_allowed": False,
        "transformer_allowed": False,
        "scale_up_training_allowed": False,
        "training_execution_allowed_in_stage18": False,
        "checkpoint_creation_allowed": False,
        "artifact_written": False,
    }
    return Stage18EvidenceBuild(dataset=dataset, report=report)


def analyze_stage18_rebuilt_evidence(
    dataset: Stage18LearningEvidenceDataset,
) -> dict[str, object]:
    """Compute after-rebuild contradiction and target-quality statistics."""

    all_actor_rows = [
        actor_row for row in dataset.rows for actor_row in row.actor_safe_view
    ]
    all_soft = [
        target
        for row in dataset.rows
        for target in row.actor_target_view["actor_soft_utility_targets"]
    ]
    all_rankings = [
        target
        for row in dataset.rows
        for target in row.actor_target_view["pairwise_ranking_targets"]
    ]
    all_hard = [
        target
        for row in dataset.rows
        for target in row.actor_target_view["hard_label_diagnostics"]
    ]
    all_critic = [
        target
        for row in dataset.rows
        for target in row.critic_target_view["edge_delta_targets"]
    ]
    signature_count = len({stage18_actor_signature_key(row) for row in all_actor_rows})
    hard_clusters = _hard_label_contradiction_clusters(dataset)
    allowed_hard = [
        target
        for target in all_hard
        if bool(target["hard_label_allowed_for_actor_training"])
    ]
    allowed_group_count = len(
        {
            stage18_actor_signature_key(actor_row)
            for row in dataset.rows
            for actor_row in row.actor_safe_view
            if _hard_allowed_for_actor_row(row, actor_row)
        }
    )
    hard_rate = (
        len(hard_clusters) / allowed_group_count if allowed_group_count else 0.0
    )
    ambiguity_counts = Counter(
        reason
        for target in all_soft
        for reason in tuple(target.get("ambiguity_reasons", ()))
    )
    high_ambiguity_downweighted = sum(
        1
        for target in all_soft
        if str(target["actor_target_ambiguity_level"]) == "high"
        and float(target["actor_edge_utility_confidence"]) < 0.5
    )
    actor_target_global_clean = all(
        actor_target_view_has_no_global_delta_fields(row.actor_target_view)
        for row in dataset.rows
    )
    critic_target_separated = all(
        target.get("target_role") == TARGET_ROLE_CRITIC_ONLY for target in all_critic
    )
    actor_safe_clean = actor_safe_view_has_no_forbidden_fields(all_actor_rows)
    primary_actor_target_contradiction_rate = _primary_actor_target_contradiction_rate(
        dataset
    )
    real_sequence_present = any(
        row.diagnostics_view.get("sequence_id") == STAGE16_SEQUENCE_ID
        for row in dataset.rows
    )
    hard_labels_not_primary = len(allowed_hard) == 0 or hard_rate <= STAGE18_HARD_LABEL_ALLOWED_CONTRADICTION_THRESHOLD
    soft_coverage = len(all_soft) / max(1, len(all_actor_rows))
    return {
        "rebuilt_sample_count": len(all_actor_rows),
        "actor_safe_signature_count": signature_count,
        "hard_label_allowed_subset_sample_count": len(allowed_hard),
        "hard_label_allowed_subset_contradiction_rate": hard_rate,
        "primary_actor_target_contradiction_rate": primary_actor_target_contradiction_rate,
        "samples_converted_to_soft_utility_target": len(all_soft),
        "soft_utility_target_coverage": soft_coverage,
        "ranking_pair_count": len(all_rankings),
        "samples_moved_to_critic_only": len(all_soft),
        "critic_only_global_target_count": len(all_critic),
        "high_ambiguity_samples_downweighted": high_ambiguity_downweighted,
        "remaining_contradiction_clusters": hard_clusters,
        "remaining_contradiction_cluster_count": len(hard_clusters),
        "dominant_remaining_ambiguity_reasons": dict(ambiguity_counts.most_common()),
        "actor_safe_view_has_no_forbidden_global_fields": actor_safe_clean,
        "actor_target_view_has_no_global_delta_fields": actor_target_global_clean,
        "critic_only_global_targets_separated": critic_target_separated,
        "hard_labels_not_primary_actor_target": hard_labels_not_primary,
        "actor_observation_signatures_still_100_percent_contradictory": (
            primary_actor_target_contradiction_rate >= 1.0
        ),
        "real_multistep_actor_safe_sequence_present": real_sequence_present,
        "stage19_gate_inputs": {
            "hard_label_allowed_contradiction_threshold": STAGE18_HARD_LABEL_ALLOWED_CONTRADICTION_THRESHOLD,
            "minimum_soft_utility_coverage": STAGE18_MIN_SOFT_UTILITY_COVERAGE,
            "minimum_ranking_pair_count": STAGE18_MIN_RANKING_PAIR_COUNT,
        },
    }


def _prepare_actor_feature_rows(
    source: LearningEvidenceDataset,
    old_causes: Mapping[tuple[object, ...], tuple[str, ...]],
) -> tuple[dict[str, object], ...]:
    previous_selected_by_key: dict[tuple[str, str, int], tuple[str, ...]] = {}
    for row in source.rows:
        sequence_id = str(row.diagnostics.get("sequence_id") or "")
        if not sequence_id:
            continue
        previous_selected_by_key[
            (sequence_id, row.topology_name, int(row.diagnostics.get("time_step", 0)))
        ] = row.selected_edges

    prepared: list[dict[str, object]] = []
    for row in source.rows:
        sequence_id = str(row.diagnostics.get("sequence_id") or "")
        time_step = int(row.diagnostics.get("time_step", 0))
        previous_key = (sequence_id, row.topology_name, time_step - 1)
        previous_selected = previous_selected_by_key.get(previous_key, ())
        actor_safe_rows = build_stage18_actor_safe_feature_rows(
            row,
            previous_selected_edges=previous_selected,
        )
        ambiguity_by_signature: dict[tuple[object, ...], tuple[str, ...]] = {}
        old_cause_lookup = _old_cause_lookup_for_row(row, old_causes)
        for actor_safe_row in actor_safe_rows:
            key = stage18_actor_signature_key(actor_safe_row)
            old_key = (
                str(actor_safe_row["agent_id"]),
                str(actor_safe_row["edge_id"]),
            )
            causes = tuple(old_cause_lookup.get(old_key, ()))
            if _is_oracle_row(row):
                causes = tuple(
                    sorted(
                        {
                            *causes,
                            ContradictionCause.ORACLE_LABEL_NOT_ACTOR_OBSERVABLE.value,
                        }
                    )
                )
            if causes:
                ambiguity_by_signature[key] = causes
        prepared.append(
            {
                "source_row": row,
                "actor_safe_rows": actor_safe_rows,
                "ambiguity_by_signature": ambiguity_by_signature,
            }
        )
    return tuple(prepared)


def _old_cause_lookup_for_row(
    row: LearningEvidenceRow,
    old_causes: Mapping[tuple[object, ...], tuple[str, ...]],
) -> dict[tuple[str, str], tuple[str, ...]]:
    lookup: dict[tuple[str, str], tuple[str, ...]] = {}
    for actor_row in row.actor_safe_rows:
        for neighbor in actor_row["local_neighbor_observations"]:
            signature = build_actor_observation_signature(actor_row, neighbor)
            if signature.key in old_causes:
                edge_id = str(getattr(neighbor, "edge_id", ""))
                lookup[(str(actor_row["agent_id"]), edge_id)] = old_causes[signature.key]
    return lookup


def _hard_label_allowed_by_signature(
    prepared: Iterable[Mapping[str, object]],
) -> dict[tuple[object, ...], bool]:
    labels_by_signature: dict[tuple[object, ...], set[str]] = defaultdict(set)
    for item in prepared:
        row = item["source_row"]
        target_by_edge = _counterfactual_targets(row.learning_targets)
        for actor_safe_row in item["actor_safe_rows"]:
            edge_id = str(actor_safe_row["edge_id"])
            if edge_id not in target_by_edge:
                continue
            key = stage18_actor_signature_key(actor_safe_row)
            labels_by_signature[key].add(str(target_by_edge[edge_id].get("action_type", "")))
    return {
        key: len(labels) <= 1 and labels != {""}
        for key, labels in labels_by_signature.items()
    }


def _diagnostics_view(
    row: LearningEvidenceRow,
    *,
    actor_safe_rows: tuple[Mapping[str, object], ...],
    actor_target_view: Mapping[str, object],
    critic_target_view: Mapping[str, object],
) -> dict[str, object]:
    forbidden_scan_passed = actor_safe_view_has_no_forbidden_fields(actor_safe_rows)
    ambiguity_reasons = Counter(
        reason
        for target in actor_target_view["actor_soft_utility_targets"]
        for reason in tuple(target.get("ambiguity_reasons", ()))
    )
    return {
        "scenario_id": row.scenario_id,
        "topology_id": row.topology_id,
        "topology_name": row.topology_name,
        "source_fixture": row.diagnostics.get("fixture_id", row.scenario_id),
        "source_dataset_id": STAGE16_DATASET_ID,
        "sequence_id": row.diagnostics.get("sequence_id"),
        "time_step": row.diagnostics.get("time_step"),
        "target_source": "stage16_edge_delta_rebuilt_into_stage18_soft_ranking_and_critic_targets",
        "ambiguity_reason": dict(ambiguity_reasons),
        "contradiction_flags": {
            "raw_stage16_hard_label_conflict_reproduced": bool(ambiguity_reasons),
            "hard_labels_primary_actor_target": False,
        },
        "forbidden_field_scan_result": {
            "actor_safe_view_passed": forbidden_scan_passed,
            "actor_target_view_has_global_delta_fields": not actor_target_view_has_no_global_delta_fields(
                actor_target_view
            ),
            "critic_target_role": critic_target_view.get("target_role"),
        },
        "feature_schema_id": STAGE18_FEATURE_SCHEMA_ID,
    }


def _hard_label_contradiction_clusters(
    dataset: Stage18LearningEvidenceDataset,
) -> tuple[dict[str, object], ...]:
    labels_by_signature: dict[tuple[object, ...], Counter[str]] = defaultdict(Counter)
    summary_by_signature: dict[tuple[object, ...], Mapping[str, object]] = {}
    for row in dataset.rows:
        hard_by_edge = {
            str(target["edge_id"]): target
            for target in row.actor_target_view["hard_label_diagnostics"]
        }
        for actor_row in row.actor_safe_view:
            target = hard_by_edge.get(str(actor_row["edge_id"]))
            if not target or not bool(target["hard_label_allowed_for_actor_training"]):
                continue
            key = stage18_actor_signature_key(actor_row)
            labels_by_signature[key][str(target["hard_label"])] += 1
            summary_by_signature[key] = _signature_summary(actor_row)
    clusters: list[dict[str, object]] = []
    for key, counter in labels_by_signature.items():
        if len(counter) <= 1:
            continue
        clusters.append(
            {
                "observation_signature_summary": dict(summary_by_signature[key]),
                "label_distribution": dict(counter),
            }
        )
    return tuple(clusters)


def _primary_actor_target_contradiction_rate(
    dataset: Stage18LearningEvidenceDataset,
) -> float:
    by_signature: dict[tuple[object, ...], list[float]] = defaultdict(list)
    target_by_row_edge: dict[tuple[int, str], Mapping[str, object]] = {}
    for row_index, row in enumerate(dataset.rows):
        for target in row.actor_target_view["actor_soft_utility_targets"]:
            target_by_row_edge[(row_index, str(target["edge_id"]))] = target
        for actor_row in row.actor_safe_view:
            target = target_by_row_edge.get((row_index, str(actor_row["edge_id"])))
            if not target:
                continue
            if float(target["actor_edge_utility_confidence"]) >= 0.5:
                by_signature[stage18_actor_signature_key(actor_row)].append(
                    float(target["actor_edge_utility_target"])
                )
    if not by_signature:
        return 0.0
    contradictory = sum(
        1 for values in by_signature.values() if max(values) - min(values) > 0.25
    )
    return contradictory / len(by_signature)


def _stage19_allowed(after: Mapping[str, object]) -> bool:
    return bool(
        after["actor_safe_view_has_no_forbidden_global_fields"]
        and after["actor_target_view_has_no_global_delta_fields"]
        and after["critic_only_global_targets_separated"]
        and after["hard_labels_not_primary_actor_target"]
        and float(after["soft_utility_target_coverage"]) >= STAGE18_MIN_SOFT_UTILITY_COVERAGE
        and int(after["ranking_pair_count"]) >= STAGE18_MIN_RANKING_PAIR_COUNT
        and not after["actor_observation_signatures_still_100_percent_contradictory"]
        and after["real_multistep_actor_safe_sequence_present"]
    )


def _hard_allowed_for_actor_row(
    row: Stage18EvidenceRow,
    actor_row: Mapping[str, object],
) -> bool:
    for target in row.actor_target_view["hard_label_diagnostics"]:
        if str(target["edge_id"]) == str(actor_row["edge_id"]) and str(
            target["agent_id"]
        ) == str(actor_row["agent_id"]):
            return bool(target["hard_label_allowed_for_actor_training"])
    return False


def _counterfactual_targets(
    raw_targets: Iterable[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    output: dict[str, Mapping[str, object]] = {}
    for target in raw_targets:
        action_type = str(target.get("action_type", ""))
        if action_type == "keep_edge":
            continue
        output[str(target.get("edge_id", ""))] = target
    return output


def _signature_summary(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "agent_id": row["agent_id"],
        "edge_id": row["edge_id"],
        "neighbor_id": row["neighbor_id"],
        "edge_active_prev": row["edge_active_prev"],
        "edge_active_current_local": row["edge_active_current_local"],
        "local_selected_edge_count": row["local_selected_edge_count"],
        "tx_budget_remaining": row["tx_budget_remaining"],
        "local_conflict_group_occupancy": row["local_conflict_group_occupancy"],
    }


def _is_oracle_row(row: LearningEvidenceRow) -> bool:
    return str(row.diagnostics.get("topology_label_role")) == "oracle_training_diagnostic_only"


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value
