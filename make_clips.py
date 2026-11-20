"""Cut the short demo recordings bundled with the app (organizers / Data team).

    python make_clips.py                       # auto-pick clips from the test users
    python make_clips.py --pick user_002:612   # add a specific (user, start-second) clip

Clips are excerpts of Meta's dataset (CC BY-NC 4.0). Keep the credit in clips/CREDITS.md.
Auto-pick scans each test user for 10 s windows with many gestures and a good mix of
types, and keeps the ones the model gets fully right plus one with a mistake.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from emg_kit.core import (cler, load_model, load_recording, predict, save_clip,
                          test_partitions)

CREDIT = """# Demo clip credits

Every `.npz` file in this folder is a short excerpt of Meta's discrete-gestures sEMG
dataset from the generic-neuromotor-interface project
(https://github.com/facebookresearch/generic-neuromotor-interface), shared under
**CC BY-NC 4.0** (non-commercial use, attribution required).

Kaifosh, P., Reardon, T. R. & CTRL-labs at Reality Labs. *A generic non-invasive
neuromotor interface for human-computer interaction.* Nature (2025).
https://doi.org/10.1038/s41586-025-09255-w

EMG is stored as float16 to keep the files small. Each clip includes a short lead-in
used only to warm up the model.

| file | user | start (s into test span) | seconds | gestures |
|---|---|---|---|---|
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--n", type=int, default=4, help="clips to auto-pick")
    p.add_argument("--pick", action="append", default=[], help="user_XXX:start_second")
    p.add_argument("--out", default="clips")
    a = p.parse_args()

    out = Path(a.out)
    model = load_model()
    parts = test_partitions()
    candidates = []  # (score, user, t0, cler)

    for user, (path, s, e) in parts.items():
        rec = load_recording(path, s, e, user)
        probs, pt = predict(model, rec.emg, rec.times)
        rel = rec.prompts.time.to_numpy() - rec.times[0]
        for t0 in np.arange(30, rec.seconds - a.seconds - 5, a.seconds):
            m = (rel >= t0) & (rel < t0 + a.seconds)
            names = rec.prompts.name[m]
            if len(names) < 6:
                continue
            w = (pt - rec.times[0] >= t0) & (pt - rec.times[0] < t0 + a.seconds)
            c = cler(probs[:, w], pt[w], rec.prompts[m])
            candidates.append((len(names) + 3 * names.nunique(), user, float(t0), c))
        print(f"{user}: scanned")

    chosen = []
    perfect = sorted([c for c in candidates if c[3] == 0], reverse=True)
    flawed = sorted([c for c in candidates if c[3] > 0], reverse=True)
    used_users = set()
    for c in perfect:  # spread clips across users
        if c[1] not in used_users and len(chosen) < a.n - 1:
            chosen.append(c)
            used_users.add(c[1])
    for c in perfect:
        if len(chosen) < a.n - 1 and c not in chosen:
            chosen.append(c)
    if flawed:
        chosen.append(flawed[0])
    for spec in a.pick:
        u, t0 = spec.split(":")
        chosen.append((0, u, float(t0), None))

    lines = []
    for _, user, t0, c in chosen:
        rec = load_recording(*parts[user], user=user)
        f = save_clip(rec, t0, a.seconds, out / f"{user}_t{int(t0):04d}.npz")
        n = int(((rec.prompts.time - rec.times[0]).between(t0, t0 + a.seconds)).sum())
        lines.append(f"| `{f.name}` | {user} | {t0:.0f} | {a.seconds:.0f} | {n} |")
        print(f"saved {f} ({f.stat().st_size/1e6:.2f} MB, {n} gestures, clip CLER {c})")
    (out / "CREDITS.md").write_text(CREDIT + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
