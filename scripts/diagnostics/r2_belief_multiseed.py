"""R2 multi-seed HONEST headline: does the belief head RECOVER current CSI, i.e. beat the stale-echo floor?

The R2 verification (Workflow wozljm9uf) found the belief head learns to ECHO its stale input (held MSE at/
above the trivial "predict the stale observed psucc" floor), so it is a no-op CSI predictor. This runs a
paired 5-seed comparison reporting, per mode, belief MSE vs the stale-echo floor (paired floor - belief;
positive => the belief RECOVERS beyond the echo) -- for BOTH the velocity-augmented and recurrent arms --
and the recurrent-vs-memoryless gap. Contract v4: >=5 seeds + CI; the conclusion is multi-seed, not pilot.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))

import csi_belief_train as bt  # noqa: E402


def _ci95(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return {"mean": round(m, 6), "lo": round(m, 6), "hi": round(m, 6), "n": n}
    sd = math.sqrt(sum((v - m) ** 2 for v in xs) / (n - 1))
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447}.get(n, 2.776)
    h = t * sd / math.sqrt(n)
    return {"mean": round(m, 6), "lo": round(m - h, 6), "hi": round(m + h, 6), "n": n}


def main() -> None:
    seeds = [0, 1, 2, 3, 4]
    report = {"scope": "R2 multi-seed: belief vs STALE-ECHO FLOOR (recovery test) + recurrent-vs-memoryless; "
                       "velocity-augmented belief head; random data; 5 seeds. paired(floor-belief)>0 => recovers.",
              "by_mode": {}}
    for mode, delay in [("delay1", 1), ("delay2", 2), ("partial", 0)]:
        floor_minus_bel, beats, rec_mse, mem_mse, paired_rm = [], 0, [], [], []
        for s in seeds:
            train = bt._scenes(mode, delay, 1000 + s, 5, 6)
            held = bt._scenes(mode, delay, 7000 + s, 5, 6)
            r = bt.train_belief(train, held, recurrent=True, epochs=80, hidden=32, seed=s, use_velocity=True)
            m = bt.train_belief(train, held, recurrent=False, epochs=80, hidden=32, seed=s, use_velocity=True)
            floor_minus_bel.append(r["stale_echo_floor_mse"] - r["held_belief_mse"])   # >0 => recovers
            beats += int(r["beats_stale_echo_floor"])
            rec_mse.append(r["held_belief_mse"]); mem_mse.append(m["held_belief_mse"])
            paired_rm.append(m["held_belief_mse"] - r["held_belief_mse"])
        report["by_mode"][mode] = {
            "paired_floor_minus_belief": _ci95(floor_minus_bel),     # >0 => belief recovers beyond stale echo
            "belief_beats_floor_seeds": f"{beats}/{len(seeds)}",
            "recurrent_mse": _ci95(rec_mse), "memoryless_mse": _ci95(mem_mse),
            "paired_memoryless_minus_recurrent": _ci95(paired_rm)}
        d = report["by_mode"][mode]["paired_floor_minus_belief"]
        print(f"  [{mode:8}] floor-belief {d['mean']:+.6f} CI[{d['lo']:+.6f},{d['hi']:+.6f}]  "
              f"beats_floor={report['by_mode'][mode]['belief_beats_floor_seeds']}  "
              f"(rec {report['by_mode'][mode]['recurrent_mse']['mean']:.5f} vs mem "
              f"{report['by_mode'][mode]['memoryless_mse']['mean']:.5f})")
    out = ROOT / "result_save" / "belief_residual" / "R2" / "belief_floor_multiseed.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
