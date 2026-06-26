# D0 — experiment_plan

**Stage:** D0 (freeze current dynamic results as the diagnostic baseline).
**Hypothesis:** N/A — D0 is a freeze + gap-analysis step, not a hypothesis test.
**Single change:** none to code; produce the gap-status doc + the frozen baseline archive.
**Controlled variables:** N/A.
**Dataset:** the existing frozen artifacts at commit `e450cb7`.
**Seeds:** the existing 0/1/2 runs (archived as-is).
**Budget:** zero new training; only file copies + doc writes.
**Expected mechanism activation:** N/A (no new run).
**Success criterion:**
- `docs/CURRENT_DYNAMIC_REPAIR_STATUS.md` exists, grounds all 10 flagged gaps in file:line evidence;
- `result_save/dynamic_baseline_frozen/README.md` records SHA, exact per-seed numbers, repro commands,
  and the binding scope labels;
- the current D4/D6 results are reproducible from the archive and labeled diagnostic-only.
**Failure criterion:** a flagged gap cannot be grounded in code (would mean the gap is mis-stated).
