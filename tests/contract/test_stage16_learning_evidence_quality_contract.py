from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage16_contract_does_not_add_training_artifact_or_model_files() -> None:
    stage16_source_files = [
        ROOT / "src" / "marl_topology" / "data" / "learning_evidence_stage16.py",
    ]
    forbidden_terms = [
        "optimizer.step",
        ".backward(",
        "torch.save",
        "write_checkpoint",
        "checkpoint_path",
        "training_loop",
        "COMA implementation",
        "Transformer implementation",
        "D:\\PhD_works\\v5",
    ]
    hits = []
    for path in stage16_source_files:
        text = path.read_text(encoding="utf-8")
        for term in forbidden_terms:
            if term in text:
                hits.append(f"{path.relative_to(ROOT)}:{term}")
    assert not hits, f"Stage 16 contract introduced forbidden implementation terms: {hits}"
