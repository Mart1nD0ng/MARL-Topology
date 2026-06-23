from pathlib import Path

from marl_topology.evaluation import build_demo_report
from marl_topology.metrics import REGISTERED_METRICS


ROOT = Path(__file__).resolve().parents[2]


def test_registered_metric_names_cover_demo_outputs() -> None:
    report = build_demo_report()
    for baseline in report["baselines"].values():
        metric_names = set(baseline) - {"topology_id", "selected_edge_ids"}
        assert metric_names <= set(REGISTERED_METRICS)


def test_metric_rows_include_registered_names_and_identifiers() -> None:
    from marl_topology.evaluation import build_demo_stack

    _, _, evaluator, _ = build_demo_stack()
    evaluation = evaluator.evaluate(set(evaluator.graph.edge_ids), topology_id="baseline:full")
    rows = evaluation.metric_rows()

    assert rows
    assert {row["metric_name"] for row in rows} == set(REGISTERED_METRICS)
    assert all(row["scenario_id"] == "demo_stage2" for row in rows)
    assert all("metric_level" in row and "used_for" in row for row in rows)


def test_src_does_not_reintroduce_old_effective_success_defaults() -> None:
    banned = ("P_eff_soft", "P_eff_hard", "hard_eval", "soft_train")
    offenders: dict[str, list[str]] = {}
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        hits = [term for term in banned if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"old effective-success defaults in src: {offenders}"


def test_demo_report_labels_full_as_baseline_only() -> None:
    report = build_demo_report()

    assert "full" in report["baselines"]
    assert report["baselines"]["full"]["topology_id"] == "baseline:full"
    assert report["oracle"]["oracle_name"] != "full"
