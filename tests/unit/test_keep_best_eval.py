"""keep_best_eval (opt-in): the policy-gradient fine-tune restores the best-validation
actor checkpoint (starting from the warm-started BC policy) before the final test, so RL
can only help, never degrade feasibility below BC. Behavior (test tau >= warm-start) is
covered by logs/verify_keep_best.py; here we pin the config contract.
"""

from marl_topology.training.production_mappo_adapter import Stage33GNNStabilityConfig


def test_keep_best_eval_defaults_off() -> None:
    assert Stage33GNNStabilityConfig().keep_best_eval is False


def test_keep_best_eval_serializes() -> None:
    payload = Stage33GNNStabilityConfig(keep_best_eval=True).to_payload()
    assert payload["keep_best_eval"] is True
    # keep-best is a checkpoint-selection policy; the frozen reward contract is untouched.
    assert payload["reward_weights_tuned"] is False
    assert payload["tau_requirement_min"] == Stage33GNNStabilityConfig().tau_requirement_min
