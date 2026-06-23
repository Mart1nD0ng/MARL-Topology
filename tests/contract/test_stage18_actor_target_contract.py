from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_stage18_report_script_is_report_only() -> None:
    text = _read("scripts/replay/stage18_evidence_rebuild_report.py")

    required = [
        "build_stage18_evidence_rebuild_report",
        "print(json.dumps",
        "report-only",
    ]
    missing = [term for term in required if term not in text]
    assert not missing

    forbidden = [
        "optimizer.step",
        ".backward(",
        "torch.save",
        "checkpoint_path",
        "write_text(",
        "write_bytes(",
        "D:\\PhD_works\\v5",
    ]
    hits = [term for term in forbidden if term in text]
    assert not hits, f"Stage 18 report script introduced forbidden terms: {hits}"

