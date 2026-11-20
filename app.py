"""Wearable AI Build Day demo: hand gestures from wrist EMG.

    python app.py            # http://127.0.0.1:7860

Demo team: everything you'll want to change is marked TODO.
Model and data: Meta, generic-neuromotor-interface (CC BY-NC 4.0).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from emg_kit.core import (CKPT_PATH, GESTURES, cler, detect_events, load_clip,  # noqa: E402
                          load_model, match_events, predict)

ROOT = Path(__file__).parent
CLIPS = sorted((ROOT / "clips").glob("*.npz"))
CHART = ROOT / "assets" / "per_user_accuracy.png"   # TODO(1:15): Data team's chart goes here

# Model: TODO(1:15) drop the Speed team's winner at models/demo_model.pt
MODEL_CANDIDATES = [os.environ.get("MODEL_PATH"), ROOT / "models" / "demo_model.pt",
                    ROOT / "models" / "baseline_state.pt", CKPT_PATH]
MODEL_PATH = next(Path(p) for p in MODEL_CANDIDATES if p and Path(p).exists())
MODEL = load_model(MODEL_PATH)
MODEL_NAME = MODEL_PATH.name

COLORS = dict(zip(GESTURES, plt.cm.tab10.colors))
ICONS = {"correct": "✅ correct", "wrong gesture": "❌ wrong gesture",
         "missed": "⚪ missed", "extra": "➕ extra"}


def plot(rec, probs, pt, events):
    t0 = rec.times[0] + rec.lead_in
    t = rec.times - t0
    show = t >= 0
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True,
                                 gridspec_kw={"height_ratios": [1.5, 1]})
    sp = 4 * float(np.std(rec.emg[:, show]))
    for ch in range(rec.emg.shape[0]):
        a1.plot(t[show], rec.emg[ch, show] + ch * sp, lw=0.5, color="0.45")
    for _, r in rec.prompts.iterrows():
        x = r.time - t0
        a1.axvline(x, color=COLORS[r["name"]], lw=1.5, alpha=0.9)
        a1.text(x, 16 * sp, r["name"].replace("_", " "), rotation=90, fontsize=8,
                va="bottom", ha="right", color=COLORS[r["name"]])
    a1.set_yticks([i * sp for i in range(16)], [f"ch{i}" for i in range(16)], fontsize=7)
    a1.set_title("16-channel wrist EMG · coloured lines = true gestures", loc="left")

    tp = pt - t0
    m = tp >= 0
    for i, g in enumerate(GESTURES):
        a2.plot(tp[m], probs[i, m], lw=1.2, color=COLORS[g], label=g.replace("_", " "))
    a2.axhline(0.35, color="k", ls=":", lw=0.8)
    for _, e in events.iterrows():
        a2.plot(e.time - t0, 1.03, "v", color=COLORS[e["name"]], ms=7, clip_on=False)
    a2.set_ylim(0, 1.08)
    a2.set_ylabel("probability")
    a2.set_xlabel("seconds")
    a2.set_title("Model output · ▼ = predicted gesture (fires above 0.35; releases with no pressed finger are ignored, as in CLER)", loc="left", fontsize=10)
    a2.legend(ncol=5, fontsize=8, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.28))
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def run(choice, upload):
    path = upload or (ROOT / "clips" / choice if choice else None)
    if not path:
        raise gr.Error("Pick a recording or upload a .npz clip.")
    rec = load_clip(path)

    t = time.perf_counter()
    probs, pt = predict(MODEL, rec.emg, rec.times)
    ms = (time.perf_counter() - t) * 1000

    events = detect_events(probs, pt)
    t0 = rec.times[0] + rec.lead_in
    events = events[events.time >= t0].reset_index(drop=True)
    table = match_events(events, rec.prompts)
    table["result"] = table["result"].map(ICONS)
    marks = table[table.predicted != "—"].rename(columns={"predicted": "name"})
    table["time"] = (table["time"] - t0).round(2)

    shown = rec.times[-1] - t0
    correct = int((table.result == ICONS["correct"]).sum())
    keep = pt >= t0
    c = cler(probs[:, keep], pt[keep], rec.prompts) if len(rec.prompts) else float("nan")
    stats = (f"### {correct} / {len(rec.prompts)} gestures recognized\n"
             f"**Model:** `{MODEL_NAME}` · **CPU time:** {ms:.0f} ms for "
             f"{rec.seconds:.1f} s of EMG ({ms / rec.seconds:.1f} ms per second of signal, "
             f"{rec.seconds * 1000 / ms:.0f}× real time) · **clip CLER:** {c:.3f} "
             f"(lower is better) · showing {shown:.0f} s from {rec.user}")
    return plot(rec, probs, pt, marks), stats, table


with gr.Blocks(title="Wearable AI: gestures from wrist EMG") as demo:
    # TODO: team names, a sentence about what you found
    gr.Markdown("""
# ✋ Reading hand gestures from wrist muscles
A wristband measures the electrical activity of your forearm muscles (**surface EMG**,
16 channels, 2,000 samples/s). A neural network turns it into 9 finger and thumb
gestures, like *index press*, *thumb swipe up* or *thumb click*, in real time.
Pick a recording and see what the model hears.
""")
    with gr.Row():
        clip = gr.Dropdown([c.name for c in CLIPS], value=CLIPS[0].name if CLIPS else None,
                           label="Recording (held-out test users)", scale=3)
        up = gr.File(label="…or upload a .npz clip", file_types=[".npz"],
                     type="filepath", scale=2)
        go = gr.Button("Recognize gestures", variant="primary", scale=1)
    stats = gr.Markdown()
    fig = gr.Plot(label="EMG and predictions")
    table = gr.Dataframe(label="Predicted vs. true", interactive=False, wrap=True)
    if CHART.exists():
        gr.Markdown("## How well does it work on people it has never seen?")
        gr.Image(str(CHART), show_label=False)
    gr.Markdown("""
---
Model, code and data: Meta, [generic-neuromotor-interface](https://github.com/facebookresearch/generic-neuromotor-interface),
**CC BY-NC 4.0** (non-commercial). Kaifosh, Reardon & CTRL-labs at Reality Labs,
*A generic non-invasive neuromotor interface for human-computer interaction*, Nature (2025).
Built at the Innovation Consulting Group Wearable AI Build Day, Dec 4.
""")
    go.click(run, [clip, up], [fig, stats, table])
    demo.load(run, [clip, up], [fig, stats, table])

if __name__ == "__main__":
    demo.launch()
