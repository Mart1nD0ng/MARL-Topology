from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_stage17_report_script_exists_and_is_report_only() -> None:
    text = _read("scripts/replay/stage17_actor_label_disambiguation_report.py")

    required = [
        "build_stage17_actor_label_disambiguation_report",
        "print(json.dumps",
        "report-only",
    ]
    missing = [term for term in required if term not in text]
    assert not missing, f"Stage 17 script missing report-only terms: {missing}"

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
    assert not hits, f"Stage 17 report script introduced forbidden terms: {hits}"


def test_stage17_source_does_not_add_training_model_checkpoint_or_v5_path() -> None:
    paths = [
        ROOT / "src" / "marl_topology" / "data" / "actor_label_disambiguation.py",
        ROOT / "scripts" / "replay" / "stage17_actor_label_disambiguation_report.py",
    ]
    forbidden = [
        "import torch",
        "optimizer.step",
        ".backward(",
        "torch.save",
        "checkpoint_path",
        "PPOTrainer",
        "MAPPOTrainer",
        "class COMA",
        "class Transformer",
        "D:\\PhD_works\\v5",
    ]
    hits = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for term in forbidden:
            if term in text:
                hits.append(f"{path.relative_to(ROOT)}:{term}")
    assert not hits, f"Stage 17 source introduced forbidden terms: {hits}"
