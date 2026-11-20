"""Score models for the Speed challenge: CLER, file size, and CPU latency.

    python speed_report.py --model baseline
    python speed_report.py --model models/int8_dynamic.pt --model models/pruned.pt --x 1.0
    python speed_report.py --model models/mine.pt --quick     # practice: ~6x faster

The baseline is always scored in the same run and the same mode, so every
comparison is apples to apples. Latency is the median ms for one prediction over
1 s of 16-channel EMG, batch 1, 1 CPU thread (see emg_kit.core.benchmark_ms).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import warnings

import torch

warnings.filterwarnings("ignore")

from emg_kit.core import (QUICK_MINUTES, all_test_users, benchmark_ms, evaluate, load_model,
                          model_size_mb, test_partitions)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", default=[],
                   help='"baseline" or a path to a .pt saved with save_whole_model. Repeatable.')
    p.add_argument("--x", type=float, default=None,
                   help="allowed CLER increase over baseline, in points (1.0 = 0.01 CLER)")
    p.add_argument("--users", nargs="*", default=None, help="limit to some test users")
    p.add_argument("--quick", action="store_true",
                   help="practice mode: 5 one-minute windows per user (~6x faster, not the official score)")
    p.add_argument("--threads", type=int, default=1, help="CPU threads for the latency test")
    p.add_argument("--out", default="results/speed_report.json")
    a = p.parse_args()

    names = ["baseline"] + [m for m in a.model if m != "baseline"]
    have, want = list(test_partitions()), all_test_users()
    print(f"Test users: {len(have)} of {len(want)} available ({', '.join(have)})")
    if a.quick:
        print(f"QUICK MODE: {QUICK_MINUTES:g} min of each user. Practice only, "
              "not the official score.")
    print(f"Latency: 1 s window, batch 1, {a.threads} thread(s)\n")

    rows = []
    for name in names:
        model = load_model(None if name == "baseline" else name)
        t = time.perf_counter()
        df = evaluate(model, users=a.users, verbose=False,
                      minutes=QUICK_MINUTES if a.quick else None)
        rows.append({
            "model": name,
            "mean_cler": float(df.cler.mean()),
            "accuracy": float(1 - df.cler.mean()),
            "size_mb": round(model_size_mb(model), 2),
            "ms_per_prediction": round(benchmark_ms(model, threads=a.threads), 3),
            "users": list(df.user),
            "per_user_cler": dict(zip(df.user, df.cler.round(5))),
            "eval_seconds": round(time.perf_counter() - t, 1),
        })
        print(f"scored {name} in {rows[-1]['eval_seconds']}s")

    base = rows[0]
    for r in rows:
        r["cler_points_vs_baseline"] = round((r["mean_cler"] - base["mean_cler"]) * 100, 3)
        r["speedup"] = round(base["ms_per_prediction"] / r["ms_per_prediction"], 2)
        r["eligible"] = None if a.x is None else r["cler_points_vs_baseline"] <= a.x

    print(f"\n{'model':<32}{'CLER':>8}{'Δpts':>8}{'MB':>8}{'ms':>9}{'x faster':>10}{'ok?':>6}")
    for r in rows:
        ok = "" if r["eligible"] is None else ("yes" if r["eligible"] else "NO")
        print(f"{Path(r['model']).name:<32}{r['mean_cler']:>8.4f}{r['cler_points_vs_baseline']:>+8.2f}"
              f"{r['size_mb']:>8.1f}{r['ms_per_prediction']:>9.2f}{r['speedup']:>10.2f}{ok:>6}")

    if a.x is not None:
        ok = [r for r in rows if r["eligible"]]          # baseline is always eligible
        w = min(ok, key=lambda r: r["ms_per_prediction"])
        if w is base:
            print(f"\nNo entry within {a.x} pts of baseline is faster than the baseline "
                  f"({base['ms_per_prediction']} ms).")
        else:
            print(f"\nWinner (within {a.x} pts of baseline): {w['model']} "
                  f"at {w['ms_per_prediction']} ms ({w['speedup']}x faster)")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"threads": a.threads, "x": a.x, "quick": a.quick,
            "torch": torch.__version__, "quant_engine": torch.backends.quantized.engine}
    out.write_text(json.dumps({"meta": meta, "results": rows}, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
