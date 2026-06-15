# Stage 21 - Objective-Stack-Aligned Assembler-Aware Rerun

Stage type: implementation-bearing evidence-generation, supervised model-rerun,
fair evaluation, and conditional policy-gradient gate.

Controlled object: Stage 21 learning evidence, actor targets, supervised
MLP/GNN rerun, same-assembler baseline comparison, and conditional
policy-gradient pilot.

Desired state: main evidence is rebuilt with the Stage 3 URLLC
finite-blocklength communication stack, Stage 4 expected-initiator PBFT
reliability, Stage 5 objective contract, and Stage 8 topology assembler. Actor
inputs remain local-only. Actor targets express assembler-aware edge priority,
not hard global oracle add/remove/keep labels. MLP and GNN are rerun on the new
target and evaluated against projected baselines under the same assembler.

## Why Stage 20 Failed

Stage 20 showed that the best full-row actor was GNN, but its
assembler-projected tau-feasible rate was `0.5593220339`, below the greedy
reliability baseline at `0.8135593220` and below the full graph baseline at
`0.6101694915`. GNN also had a projection rejection rate of `0.6647058824`.

The immediate Stage 20 conclusion was actor score and assembler projection
mismatch. Stage 21 treats that as incomplete. The deeper risk is that Stage
16-20 learning evidence still flowed from the early `SimpleLinkModel` plus
minimal `TopologyEvaluator` min-link abstraction, while the final project
objective stack already has Stage 3 finite-blocklength communication records,
Stage 4 expected-initiator PBFT reliability, and Stage 5 reliability/latency
/energy objective governance.

## Why Evidence Is Rebuilt

The actor can only learn the objective it is shown. If evidence rows and edge
targets are generated from an early evaluator, supervised losses can improve
while deployment quality remains misaligned with the final communication and
consensus objective. Stage 21 therefore audits the Stage 16-20 lineage and
builds new rows with explicit evaluator identifiers:

- Stage 3 finite-blocklength link and network communication records.
- Deadline retransmission, scheduled latency, successful delivery latency, and
  energy accounting.
- Stage 4 expected-initiator PBFT reliability with per-primary reliability.
- Stage 5 tau requirement `tau_requirement_min = 0.9`, where reliability is a
  requirement and latency/energy remain objectives.

If the final stack cannot be used, Stage 21 must mark the evaluator as partial
or blocked and must not run policy-gradient.

## Why The Same Assembler Is Required

Stage 20 compared actor outputs after assembler projection with baselines that
were not all projected through the same assembler constraints. That can make a
raw greedy or raw full graph baseline look better than a deployment actor that
must obey tx/rx budgets, duplicate-edge rejection, role validity, and conflict
rules.

Stage 21 uses raw baselines as diagnostics only. The main comparison is:

- `full_graph_projected`
- `greedy_reliability_projected`
- `random_projected`
- `mlp_actor_projected`
- `gnn_actor_projected`

All main rows pass through the same deployment assembler family and the same
objective-stack evaluator.

## Why Targets Are Assembler-Aware

Hard add/remove/keep labels can encode global topology state and oracle effects
that a deployment actor cannot observe. Stage 21 instead trains actor edge
scorers against:

- soft edge utility targets;
- pairwise local rankings;
- low-priority or abstain signals for invalid, budget-rejected,
  conflict-rejected, redundant, and high-cost unselected edges.

Global edge-delta counterfactuals remain critic-only. Oracle rows remain
diagnostic and are not deployment actor input.

## Why Policy-Gradient Is Conditional

Policy-gradient can amplify the same mismatch if the supervised actor still
proposes edges that the assembler rejects or if the evidence still uses the
wrong evaluator stack. Stage 21 therefore runs policy-gradient only after hard
gates pass for objective-stack alignment, target quality, fair same-assembler
evaluation, projection mismatch improvement, actor performance, and boundary
safety.

If any hard gate fails, Stage 21 stops, writes a root-cause review, updates
project state to blocked awaiting owner decision, and recommends a specific
repair stage.

## Pass / Fail Meaning

Pass means the evidence, target, supervised rerun, and fair projected baseline
comparison are coherent enough for a small controlled policy-gradient pilot
inside Stage 21. It does not authorize Stage 22 scale-up or final architecture
selection.

Fail means at least one control loop sensor says the learning stack is still
misaligned. In that case, policy-gradient is forbidden and the next step must
repair the failing layer: evidence stack, target construction, assembler
constraints, actor features, or baseline fairness.

## Verification

Required validation after each major part:

```powershell
python -m pytest -q
python harness\scripts\validate_tasks.py
```
