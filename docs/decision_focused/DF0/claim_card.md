# DF0 — Claim Card

**Claim (DF0, framing/design only):** The Decision-Focused campaign is correctly framed and its validation design
for goals 1 and 2 is provably sound *before any experiment* — specifically: (i) goal 2 is a **decorrelation-distance
knob** on the already-shadow-faded channel (not "enable shadowing"), and its honesty rests on five named guards
(leak-free R² gate, fair paired baseline, monotone-ρ knob proof, recovered-fraction metric, geometry-null primary
endpoint); (ii) goal 1's decision-focused objective (**BWAR**) is validated on the **deployed** anchor feasibility
(not MSE), with the true-CSI oracle as ceiling and named fake-positive guards; (iii) goal 3's softmax question is
resolved (category error for per-edge logits; edit-selection is the right place for softmax).

**This card covers which path:** the DESIGN/VALIDATION path only. DF0 asserts **no empirical result** about
feasibility, recoverability, or conversion — those are DF1–DF5 with their own cards.

**Evidence class:** code-verified facts (anchor structure `dynamic_baselines.py:48-76`; shadowing-on
`dynamic_rl.py:562` + `build_operating_point_dataset.py:57`; leak vectors `csi_observation_model.py:37-39`,
`dynamic_frames.py:259`) + a 3-lens adversarial research/red-team pass. NO training run, NO seeds — because DF0
makes no empirical headline (that is exactly why a Claim Card without seeds is legitimate here).

**Falsifiable predictions this design commits to (checked in DF1–DF2):**
- The leak-free R² test will be runnable and will discriminate (R²(shadow_t | leak-free) is *measurable*; the
  design is void if current distance already determines current shadow, i.e. R² ≥ 0.05 — that would be a real leak
  to fix, and the test is designed to catch it).
- `ρ(d_corr)` is monotone non-decreasing with `ρ(100) − ρ(10) > 0.1` (else the knob claim is false and the sweep
  is meaningless — the design says say-so, don't proceed).
- Byte-identical at `d_corr = 10 m` default.

**What would INVALIDATE the DF0 design (and force a revise):** any of — the leak-free test is not actually
failing-first / not load-bearing; the `RGF` denominator is ~0 so the metric is undefined at the knobs of interest;
the anchor used in the BWAR loss is not byte-identical to the deployed anchor; the geometry-null arm cannot be
constructed (so temporal-structure recovery cannot be isolated from observing current distance).

**No-Silent-Citation (§8):** every cited fact above has a file:line source in `validation_design.md` / this card;
the three research lenses' raw outputs are the DF0 research artifacts (agent transcripts in the session task dir).
