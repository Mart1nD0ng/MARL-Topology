"""T6 (Temporal Recovery) — env temporal-structure diagnostic (task-1 fallback).

Pins the diagnostic logic: the R^2 fit recovers a linear signal and rejects noise, and the synthetic AR sweep
shows recovery INCREASES with the temporal autocorrelation rho (validating that the method recovers
temporally-autocorrelated structure when it is present, and recovers nothing at rho=0). Fails on HEAD: the T6
generator does not exist yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT / "scripts" / "train", _ROOT / "scripts" / "diagnostics"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def test_fit_r2_recovers_linear_signal_and_rejects_noise() -> None:
    import t6_env_temporal_structure_gen as g6t
    torch.manual_seed(0)
    X = torch.randn(400, 2)
    y = X[:, 0] * 1.5 - 0.5                                          # linearly recoverable from X
    r2 = g6t._fit_r2(X[:200], y[:200], X[200:], y[200:], 0)
    assert r2 > 0.9
    yn = torch.randn(400)                                           # pure noise, unrelated to X
    r2n = g6t._fit_r2(X[:200], yn[:200], X[200:], yn[200:], 0)
    assert r2n < 0.2                                                # cannot predict noise from X


def test_synth_ar_recovery_increases_with_rho() -> None:
    """Absolute recoverability r2_pred rises with the temporal autocorrelation: rho=0 (IID) -> ~0 (unrecoverable);
    rho=0.9 -> much higher. This is the knob task-1's env temporal features would add."""
    import t6_env_temporal_structure_gen as g6t
    r_lo = g6t._synth_ar(0.0, 0)
    r_hi = g6t._synth_ar(0.9, 0)
    assert r_lo["r2_pred"] < 0.15                                    # rho=0: current unrecoverable from stale
    assert r_hi["r2_pred"] > r_lo["r2_pred"] + 0.3                   # rho=0.9: much more recoverable
