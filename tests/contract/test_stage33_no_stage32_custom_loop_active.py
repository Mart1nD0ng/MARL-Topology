from pathlib import Path

from marl_topology.training.production_mappo_adapter import Stage33ProductionMappoAdapter
from marl_topology.training.stage32_production_training import (
    STAGE32_CUSTOM_LOOP_ACTIVE_PRODUCTION_PATH,
    STAGE32_CUSTOM_LOOP_STATUS,
)


ROOT = Path(__file__).resolve().parents[2]


def test_stage32_custom_loop_is_archived_not_active_production_path() -> None:
    assert STAGE32_CUSTOM_LOOP_ACTIVE_PRODUCTION_PATH is False
    assert STAGE32_CUSTOM_LOOP_STATUS == "archived_inactive_reproducibility_only_stage33"


def test_stage33_adapter_does_not_import_or_call_stage32_training_loop() -> None:
    adapter_text = (
        ROOT / "src" / "marl_topology" / "training" / "production_mappo_adapter.py"
    ).read_text(encoding="utf-8")
    script_text = (
        ROOT / "scripts" / "train" / "stage33_gnn_stability_repair_training.py"
    ).read_text(encoding="utf-8")

    assert "from marl_topology.training.stage32_production_training" not in adapter_text
    assert "import marl_topology.training.stage32_production_training" not in adapter_text
    assert "train_production(" not in adapter_text
    assert "train_production(" not in script_text
    assert (
        Stage33ProductionMappoAdapter().official_component_report()[
            "stage32_custom_training_loop_called"
        ]
        is False
    )


def test_stage32_legacy_script_requires_explicit_inactive_flag() -> None:
    text = (
        ROOT / "scripts" / "train" / "stage32_production_training_run.py"
    ).read_text(encoding="utf-8")

    assert "--allow-inactive-stage32-legacy-loop" in text
    assert "Stage32 custom training is archived and inactive for production" in text
    assert "stage33_gnn_stability_repair_training.py" in text
