"""Phase 7 D1: the Graph-MAPPO critic + update modules never leak into the deployed path.

The centralized critic and the PPO update live under the CTDE training subtrees (models/, training/);
no deployed module (protocol/ except the spec-mandated torch quorum tail, policies/, data/,
evaluation/, the package root) may import them, and the deployment-purity gates still pass with them
present.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "marl_topology"


def test_critic_and_update_live_in_exempt_training_subtrees():
    assert (SRC / "models" / "centralized_graph_critic.py").exists()
    assert (SRC / "training" / "graph_mappo.py").exists()


def test_no_deployed_module_imports_the_critic_or_update():
    deployed = []
    for sub in ("protocol", "policies", "data", "evaluation"):
        deployed += list((SRC / sub).rglob("*.py"))
    deployed += list(SRC.glob("*.py"))  # package root
    banned = ("centralized_graph_critic", "graph_mappo", "CentralizedGraphCritic",
              # Phase 8b: the action-conditioned Q critic + COMA counterfactual credit are training-only
              "counterfactual_credit", "counterfactual_advantages", "critic_q_value", "forward_q")
    offenders = {}
    for path in deployed:
        text = path.read_text(encoding="utf-8")
        hits = [b for b in banned if b in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits
    assert not offenders, f"deployed path references the training-only critic/update: {offenders}"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deployment_purity_gates_pass_with_critic_present():
    # the critic + graph_mappo modules already exist on disk; the deployed-path scans must still pass
    g80 = _load("g80", "tests/contract/test_stage8_0_actor_policy_interface_contract.py")
    g90 = _load("g90", "tests/contract/test_stage9_0_local_mlp_edge_scorer_contract.py")
    g8 = _load("g8", "tests/contract/test_stage8_policy_architecture_and_assembler_contract.py")
    g80.test_stage8_0_source_keeps_training_model_checkpoint_v5_and_legacy_metric_out()
    g90.test_stage9_0_source_scan_allows_only_local_mlp_torch_model()
    g8.test_stage8_source_scan_blocks_model_training_and_legacy_migration()
