import pytest

from marl_topology.training.policy_gradient.pilot_runner import (
    STAGE23_PASS_VERDICT,
    Stage23PolicyGradientConfig,
    run_stage23_selected_physical_policy_gradient_pilot,
)
from marl_topology.training.policy_gradient.samplers import (
    ACTIVE_POLICY_GRADIENT_SAMPLER_ID,
    PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID,
)


@pytest.mark.xfail(
    reason=(
        "Superseded scaffold gate. Stage 23 is a deprecated REINFORCE A/B/C sampler "
        "micro-pilot; the PRODUCTION decode path is decentralized local mutual "
        "acceptance (policies/decentralized_mutual_acceptance.py) on the K-round "
        "message-passing actor + critic-planner/BC recipe, NOT the Plackett-Luce / "
        "Bernoulli / endpoint REINFORCE samplers this pilot ranks. Under the "
        "owner-approved feasibility-first BARRIER reward (Stage 31) the 3-scene "
        "tie-break now promotes the Bernoulli sampler instead of the pre-registered "
        "Plackett-Luce one (all three pass safety), so pass_gate flips to False. This "
        "is a stale pre-registration in a dead code path, not a regression. Kept xfail "
        "(machinery still exercised by the sibling record-fields test) pending removal "
        "of the Stage-23 PG scaffold."
    ),
    strict=False,
)
def test_stage23_selected_physical_policy_gradient_pilot_passes_micro_gate() -> None:
    report = run_stage23_selected_physical_policy_gradient_pilot(
        config=Stage23PolicyGradientConfig(policy_updates=1),
        project_root=".",
    )

    assert report["verdict"] == STAGE23_PASS_VERDICT
    assert report["pass_gate"] is True
    assert report["selected_action_semantics"] == "undirected_physical_link_v1"
    assert report["samplers_tested"] == [
        "endpoint_budgeted_physical_proposal_sampler",
        "physical_bernoulli_proposal_sampler",
        "physical_plackett_luce_top_k_sampler",
    ]
    assert report["sampler_selection"]["selected_sampler_id"] == PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID
    assert report["active_policy_gradient_sampler_id"] == ACTIVE_POLICY_GRADIENT_SAMPLER_ID
    assert report["reward_config_unchanged"] is True
    assert report["reward_weight_tuning_performed"] is False
    assert report["checkpoint_written"] is False
    assert report["artifact_written"] is False
    assert report["v5_modified"] is False


def test_stage23_pilot_records_required_projection_and_objective_fields() -> None:
    report = run_stage23_selected_physical_policy_gradient_pilot(
        config=Stage23PolicyGradientConfig(policy_updates=1),
        project_root=".",
    )
    winner = report["sampler_selection"]["selected_sampler_id"]
    record = report["sampler_reports"][winner]["representative_records"][0]

    required_fields = {
        "pre_projection_proposals",
        "proposal_logprob",
        "proposal_entropy",
        "post_projection_selected_physical_edges",
        "projection_rejection_reasons",
        "top_proposal_rejection_rate",
        "above_threshold_rejection_rate",
        "rejection_by_reason",
        "selected_edge_count",
        "reward_surrogate",
        "consensus_success_probability",
        "latency",
        "energy",
        "tau_feasible",
        "violation_indicator",
    }
    assert required_fields <= set(record)
    assert record["policy_action"] == "proposal"
    assert record["environment_transition"] == "projected_topology"
