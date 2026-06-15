# Stage 34 — 7×5 Portable Rerun & Resource Requirements

How to re-run the Stage 34 **seven-ablation × five-seed** GNN production sweep on
an adequately sized machine, and why it cannot run on a small one. Prepared after
the run failed on a 34 GB / 16-core dev box (decision: do not run there; run on a
≥64 GB host instead).

## What the run is

`run_stage34_7x5` drives the official MAPPO loop
(`Stage33ProductionMappoAdapter._run_seed` → rollout + GAE + clipped policy/value
loss + KL early-stop) for the seven Stage 34 GNN ablations
(`v2_reference`, `v2_plus_residual`, `v2_plus_norm`, `v2_plus_role_features`,
`v2_plus_resource_features`, `v2_plus_depth_2_3`, `v3_current`) across five seeds
(`3401–3405`) = **35 independent runs**, on the materialized real-family
graph-necessity dataset (7 families × 72 = 504 scenarios; split 350/77/77;
nodes 6–10; duplicate-rate 0; zero leakage). No model weights are written.

Protocol: `low_lr_with_warmup`, train/eval/test 350/77/77, rollout_steps 16,
transitions/update 5600, minibatch 64, 4 epochs, max_updates 20, eval_every 5,
tau_requirement_min 0.9.

## Why it failed on the small machine (root cause)

The other-device report
(`result_save/stage34_gnn_ablation_diagnostics/stage34_production_run/training_report.json`)
shows all 35 runs returning
`BrokenProcessPool('A process in the process pool was terminated abruptly …')`,
i.e. **every pool worker was killed (OOM)**, not a training collapse. The
canonical runner hardcodes `MAX_WORKERS = 14`, and each worker holds the full
train/eval/test row-contexts + critic at **~8.7 GB resident**, so 14 workers need
**~122 GB**. Any machine with materially less RAM kills workers on launch.

## Measured resource profile (dev box: 16 logical CPU, 34 GB RAM, torch 2.11+cu126)

| Quantity | Value | Notes |
|---|---|---|
| Parent dataset build (once) | ~152 s | materialize 504 scenarios + train shared critic |
| Per-worker resident memory | **~8.7 GB** | 504 evaluator-backed contexts + critic + torch |
| Per-update wall time | **~395 s** | 5,600 transitions/update; CPU-bound objective-stack evaluator |
| Per-run wall time (20 updates) | **~131 min** | one (ablation, seed) run |
| Total compute | **~76 core-hours** | 35 runs × ~131 min |

Per-worker **memory** is data-driven and fairly host-independent (use 8.7 GB for
sizing). Per-update **time** is CPU-dependent, so the ETAs below are approximate;
re-measure on the target host with `logs/stage34_resource_probe.py` if needed.

The bottleneck is the **CPU** Stage-21 objective-stack evaluator (finite-blocklength
links + shared-spectrum interference + expected-initiator PBFT) called per
transition — **not** neural matmul (the edge-scorer GNN/MLP are tiny). A high-core,
high-RAM CPU box beats a GPU box here; **no GPU is required**.

## Worker sizing

```
workers = min( floor((RAM_GB - 6) / 8.7),        # memory cap (binding ≤ ~128 GB)
               floor(physical_cores / threads),   # CPU cap (threads default 2)
               35 )                                # only 35 jobs exist
wall_time ≈ 131 min × 35 / workers
```

| Host RAM | Cores | Mem cap | CPU cap | Workers | Approx wall time |
|---|---|---|---|---|---|
| 32 GB | 8 | 2 | 4 | 2 | ~38 h |
| 48 GB | 8 | 4 | 4 | 4 | ~19 h |
| 64 GB | 16 | 6 | 8 | **6** | **~12.7 h** |
| 96 GB | 16 | 10 | 8 | 8 | ~9.6 h |
| 128 GB | 32 | 14 | 16 | **14** | **~5.5 h** ← original design point |
| 192 GB | 32 | 21 | 16 | 16 | ~4.8 h |
| 256 GB | 64 | 28 | 32 | 28 | ~2.7 h |

**Practical minimum: 64 GB / 16-core (~13 h). Comfortable target: 128 GB / 32-core
(~5.5 h)** — which is what the hardcoded `MAX_WORKERS=14` implicitly assumed.

## How to run (auto-sized, no manual tuning)

`scripts/train/stage34_production_run_sized.py` detects RAM + cores and picks a
memory-safe worker count. Override via env vars if you want.

```bash
# Recommended: auto-size to the host
python scripts/train/stage34_production_run_sized.py

# Force a worker count (bash / Linux cloud)
STAGE34_MAX_WORKERS=14 STAGE34_THREADS_PER_WORKER=2 \
  python scripts/train/stage34_production_run_sized.py

# Long run in the background (Linux)
nohup python scripts/train/stage34_production_run_sized.py > stage34_sized.log 2>&1 &
```

```powershell
# Force a worker count (Windows PowerShell)
$env:STAGE34_MAX_WORKERS = 14
python scripts/train/stage34_production_run_sized.py

# Optional: re-measure per-worker memory / per-update time on this host first
python logs/stage34_resource_probe.py
```

Env knobs: `STAGE34_MAX_WORKERS`, `STAGE34_THREADS_PER_WORKER` (default 2),
`STAGE34_SAMPLES_PER_FAMILY` (default 72 → 504 scenarios),
`STAGE34_PER_WORKER_GB` (default 8.7), `STAGE34_MAX_UPDATES` (default 20),
`CUDA_VISIBLE_DEVICES` (default `""` = CPU-only; set `0` to opt back into GPU).

Recommended sanity check before the multi-hour run — a fast reduced pass that
proves the worker pool is healthy on this host (look for stop reasons that are
**not** `worker_error`):

```bash
STAGE34_MAX_UPDATES=1 STAGE34_MAX_WORKERS=8 \
  python scripts/train/stage34_production_run_sized.py
```

The canonical `scripts/train/stage34_production_run.py` runs the identical
protocol but with a fixed `MAX_WORKERS = 14` **and** the default `fork` start
method — only safe on a ~128 GB host, and it fork-crashes on Linux (see below).
The sized runner forces `spawn`.

## Troubleshooting: `BrokenProcessPool` on Linux (even with huge RAM)

Symptom: the parent builds inputs fine (`split_counts=... f4_consistent=True`),
then **all 35 runs finish instantly** with `stop=worker_error`
(`BrokenProcessPool`), regardless of available RAM.

Cause: this is **not** OOM. `ProcessPoolExecutor` defaults to the `fork` start
method on Linux. The parent has already initialized torch's MKL/OpenMP
threadpools (and, on a GPU box, a CUDA context) while training the shared critic.
Forking that native state leaves the children broken; they crash on their first
torch op, which breaks the pool and cascades `worker_error` across every job.
Windows never hit this because it always uses `spawn`.

Fix (already applied in `stage34_production_run_sized.py`):
- `multiprocessing.set_start_method("spawn", force=True)` — clean workers.
- `CUDA_VISIBLE_DEVICES=""` by default — CPU-only, removing the CUDA-fork hazard
  (no GPU benefit here anyway).

Observed on a 679 GB / 224-core AutoDL box: with `fork`, all 35 runs returned
`worker_error` instantly. `spawn` is the same start method that runs cleanly on
the Windows dev box, so it is the expected fix; confirm on the server with the
sanity check above. Note `spawn` re-pickles the dataset + critic to each worker,
so the first ~minute is worker startup before training output appears.

### Second failure mode: thread explosion on a many-core host

Symptom: with `spawn` already in place, **8 workers succeed but 35 crash** at
startup with `worker_error`, while `/dev/shm` (e.g. 45 GB) and `ulimit -n`
(e.g. 1048576) are both ample — so it is **not** shm/FD/RAM exhaustion.

Cause: BLAS/OpenMP (MKL, OpenBLAS) default to **one thread per core**. On a
224-core box each worker spins up ~224 threads; 35 workers ≈ **7,800 threads**,
which exceeds the container's process/thread limit (cgroup `pids.max`, `ulimit
-u`) and kills workers. 8 workers (~1,800 threads) stayed under it.
`torch.set_num_threads(2)` is too late — the BLAS pools are sized at import.

Fix (already applied in `stage34_production_run_sized.py`): set
`OMP_NUM_THREADS = MKL_NUM_THREADS = OPENBLAS_NUM_THREADS = NUMEXPR_NUM_THREADS =
VECLIB_MAXIMUM_THREADS = STAGE34_THREADS_PER_WORKER` (default 2) **before** torch
import, so total threads ≈ workers × 2 (e.g. 70) regardless of core count. The
run logs the cap as `OMP_NUM_THREADS=...`. Confirm the host limit with
`ulimit -u` and `cat /sys/fs/cgroup/pids.max` (cgroup v2) if a crash persists.

## Outputs

Writes to a dedicated dir so existing reports are preserved as evidence:

```
result_save/stage34_gnn_ablation_diagnostics/stage34_production_run_sized/
  ├── manifest.json          # Stage 5.10-validated dry-run manifest
  └── training_report.json   # per-ablation collapse_rate, test/eval tau, selection gate
```

Untouched: `…/stage34_production_run/` (other-device **failed** report) and the
canonical runner's output. No model checkpoints are written (no-checkpoint
discipline). tau stays 0.9; MLP is not promoted; no reward tuning — protocol
unchanged.

## Pass criteria (unchanged from the Stage 34 gate)

A GNN is selectable only if `collapse_rate ≤ 0.20`; among those, the highest
`mean_test_tau_feasible` wins. The gate passes only if all seven ablations ran the
full protocol, a stable winner is selected, and MLP is not promoted. Otherwise the
report records `owner_decision_required: true`.

## Optional future optimization (not required for the rerun)

Per-worker 8.7 GB is dominated by 504 **read-only** evaluator-backed row-contexts
rebuilt inside every worker. Building them once and sharing them (shared memory /
`fork` copy-on-write on Linux) would cut per-worker memory sharply and let a 64 GB
box use many more workers. Out of scope here; noted for later.
