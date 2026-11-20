"""Regenerate the three starter notebooks. Organizers only.

    python tools/build_notebooks.py
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent

SETUP = r'''# --- Setup (run once) -------------------------------------------------------
# Local: you've already run `pip install -r requirements.txt` in this repo, so skip ahead.
# Colab: this cell clones the repo, installs it, and points at the shared data.
import os, sys
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    REPO_URL = "https://github.com/sandeepvijayarao09/wearable-ai-build-day.git"  # organizers fill in
    if not os.path.exists("/content/repo"):
        !git clone -q {REPO_URL} /content/repo
    %cd /content/repo
    !pip install -q -r requirements.txt
    # Shared data: in Drive, right-click the shared "emg_data" folder ->
    # Organize -> Add shortcut -> My Drive. Then:
    from google.colab import drive
    drive.mount("/content/drive")
    os.environ["EMG_DATA_DIR"] = "/content/drive/MyDrive/emg_data"
    os.environ["EMG_MODELS_DIR"] = "/content/drive/MyDrive/emg_data/emg_models"
else:
    # Run from the repo root so `emg_kit` imports.
    if not os.path.exists("emg_kit") and os.path.exists("../emg_kit"):
        os.chdir("..")
sys.path.insert(0, os.getcwd())'''


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    nbf.write(nb, ROOT / name)
    print("wrote", name)


# ---------------------------------------------------------------- 00 warm-up
write("00_plot_emg.ipynb", [
    md("""
# 00 · Warm-up: what does wrist EMG look like?

**10 minutes, everyone.** You'll load one recording from Meta's dataset and plot the
16 EMG channels with the gesture labels on top.

Surface EMG picks up the tiny electrical signals your forearm muscles make when they
fire. The wristband has 16 electrodes around the wrist and samples each one 2,000
times a second. When you press your index finger or swipe your thumb, a different mix
of channels lights up. That pattern is what the model learns.

*Data: Meta, generic-neuromotor-interface (CC BY-NC 4.0). Cite the Nature 2025 paper.*
"""),
    code(SETUP),
    code("""
import matplotlib.pyplot as plt
import numpy as np
from emg_kit.core import DATA_DIR, GESTURES, load_recording, test_partitions

parts = test_partitions()
print("Test users available here:", list(parts))
user = list(parts)[0]
path, start, end = parts[user]
rec = load_recording(path, start, end, user)
print(f"{user}: {rec.emg.shape[0]} channels x {rec.emg.shape[1]:,} samples "
      f"= {rec.seconds/60:.1f} minutes, {len(rec.prompts)} labelled gestures")
"""),
    md("""
The labels are just a table of *which gesture* happened *when*. Have a look:
"""),
    code("""
rec.prompts.head(10)
"""),
    code("""
rec.prompts["name"].value_counts()
"""),
    md("""
## Plot 6 seconds

Each grey line is one electrode, stacked so they don't overlap. Black lines mark the
moments a gesture was performed.
"""),
    code("""
T0 = 60.0      # seconds into the recording -- try changing this
SPAN = 6.0     # seconds to show

t = rec.times - rec.times[0]
win = (t >= T0) & (t < T0 + SPAN)

fig, ax = plt.subplots(figsize=(13, 6))
spacing = 4 * np.std(rec.emg[:, win])
for ch in range(rec.emg.shape[0]):
    ax.plot(t[win], rec.emg[ch, win] + ch * spacing, lw=0.6, color="0.45")

labels = rec.prompts[(rec.prompts.time - rec.times[0]).between(T0, T0 + SPAN)]
for _, row in labels.iterrows():
    x = row.time - rec.times[0]
    ax.axvline(x, color="k", lw=1)
    ax.text(x, 16 * spacing, row["name"], rotation=90, va="bottom", ha="right", fontsize=9)

ax.set_yticks([i * spacing for i in range(16)], [f"ch{i}" for i in range(16)], fontsize=7)
ax.set_xlabel("time (s)")
ax.set_title(f"{user}: 16-channel wrist EMG with gesture labels")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
"""),
    md("""
**Try it:** change `T0` to find a `thumb_click`. Which channels move the most? Now find
an `index_press`. Is it the same channels?

That's the whole problem: different gestures, different patterns across 16 channels.
Next, the Data team runs Meta's model on this ->
`01_evaluate.ipynb`.
"""),
])

# ---------------------------------------------------------------- 01 evaluate
write("01_evaluate.ipynb", [
    md("""
# 01 · Data team: reproduce the baseline

Goal: run Meta's pretrained model on the **test users**, get **CLER** per user, and
make a chart for the demo. Your mean CLER should match the baseline in the README.
If it does, the pipeline works.

**CLER** (classification error rate) is Meta's metric for this task. Every gesture
the model fires is lined up in time with the true gestures. CLER is the share that
got the wrong name, averaged over the 9 gesture types. **Lower is better.**
*Gesture accuracy = 1 − CLER.*

*Model and data: Meta, generic-neuromotor-interface (CC BY-NC 4.0).*
"""),
    code(SETUP),
    code("""
import matplotlib.pyplot as plt
import pandas as pd
from emg_kit.core import (all_test_users, detect_events, evaluate, load_model,
                          load_recording, predict, test_partitions)

model = load_model()   # Meta's checkpoint from ~/emg_models (or $EMG_MODELS_DIR)
print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

have, want = list(test_partitions()), all_test_users()
print(f"{len(have)} of {len(want)} test users on this machine: {have}")
if len(have) < len(want):
    print("Point EMG_DATA_DIR at the full shared dataset for all 10.")
"""),
    md("""
## 1. One user, step by step

Before running everyone, look at what the model actually outputs for one recording.
"""),
    code("""
user = have[0]
rec = load_recording(*test_partitions()[user], user=user)
probs, prob_times = predict(model, rec.emg, rec.times)
print("EMG in:   ", rec.emg.shape, "(channels, samples at 2 kHz)")
print("probs out:", probs.shape, "(gestures, steps at 200 Hz)")

events = detect_events(probs, prob_times)   # threshold 0.35, 50 ms debounce
print(f"model fired {len(events)} gestures; there were {len(rec.prompts)} real ones")
events.head()
"""),
    code("""
from emg_kit.core import GESTURES
T0, SPAN = 60, 8
t = prob_times - rec.times[0]
w = (t >= T0) & (t < T0 + SPAN)
fig, ax = plt.subplots(figsize=(13, 4.5))
for i, g in enumerate(GESTURES):
    ax.plot(t[w], probs[i, w] + i, lw=1, label=g)
for _, r in rec.prompts.iterrows():
    x = r.time - rec.times[0]
    if T0 <= x < T0 + SPAN:
        i = GESTURES.index(r["name"])
        ax.plot([x, x], [i, i + 1], color="k", lw=2)
ax.set_yticks([i + 0.5 for i in range(9)], GESTURES)
ax.set_xlabel("time (s)")
ax.set_title("Predicted probability per gesture (colour) vs. true gesture (black tick)")
plt.tight_layout()
"""),
    md("""
## 2. All test users -> the baseline number

Each test user is 25-42 minutes of EMG. Expect ~30 s per user on a fast laptop and
~5 minutes per user on Colab's CPU. On Colab, run the 3 users in the small subset
(target 0.1197) and leave all 10 for a laptop.
"""),
    code("""
results = evaluate(model)
results.to_csv("results_baseline.csv", index=False)
print(f"\\nMEAN CLER over {len(results)} users: {results.cler.mean():.4f}"
      f"   (gesture accuracy {1 - results.cler.mean():.1%})")
results
"""),
    md("""
## 3. Per-user chart (goes into the demo at 1:15)
"""),
    code("""
r = results.sort_values("accuracy")
fig, ax = plt.subplots(figsize=(8, 0.45 * len(r) + 1.2))
ax.barh(r.user, r.accuracy * 100, color="#4C6EF5")
ax.axvline(results.accuracy.mean() * 100, color="k", ls="--", lw=1,
           label=f"mean {results.accuracy.mean():.1%}")
for y, v in zip(r.user, r.accuracy):
    ax.text(v * 100 + 0.3, y, f"{v:.1%}", va="center", fontsize=9)
ax.set_xlabel("gesture accuracy (1 − CLER), %")
ax.set_xlim(max(0, r.accuracy.min() * 100 - 10), 100)
ax.set_title("Meta's pretrained model, held-out test users")
ax.legend(loc="lower right", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("per_user_accuracy.png", dpi=160)
print("saved per_user_accuracy.png -> hand this to the Demo team")
"""),
    md("""
## Stretch goals

- **Which gestures are hardest?** Use `generic_neuromotor_interface.cler.compute_confusion_matrix`
  on one user and plot it as a heatmap.
- **Pick demo clips.** Find 5–10 second windows where the model gets every gesture right
  (and one where it doesn't, which is honest and more interesting). Send `(user, start second)`
  to the Demo team. `make_clips.py` shows how clips are cut.
- **Threshold sweep.** CLER uses a 0.35 threshold. What happens at 0.25 or 0.5?
"""),
])

# ---------------------------------------------------------------- 02 speed
write("02_speed.ipynb", [
    md("""
# 02 · Speed team: smaller and faster, same accuracy

The model has to run on a laptop CPU (and one day on a wristband). Your job is to make
it **faster** and **smaller** without losing accuracy.

**Challenge rule:** the fastest model by `speed_report.py` wins, as long as its mean
CLER is within **X points** of the baseline (X announced at 10:30). Latency = median
ms to process **1 second of EMG**, batch 1, **1 CPU thread**.

We start with the classic first move: **INT8 dynamic quantization**. LSTM and Linear
weights go from 32-bit floats to 8-bit integers, about 3.6x smaller. Whether it's
*faster* depends on your CPU. On Intel/AMD (and Colab), PyTorch uses the `fbgemm`
backend: in our test on Colab, INT8 was **2.4x faster** (145 -> 60 ms). On Apple Silicon
it uses `qnnpack`, and on an M5 Pro the INT8 LSTM was **8x slower** than fp32
(12.5 -> 99 ms). Measure; don't assume.

*Model: Meta, generic-neuromotor-interface (CC BY-NC 4.0).*
"""),
    code(SETUP),
    code("""
import pandas as pd
from emg_kit.core import (benchmark_ms, evaluate, load_model, model_size_mb,
                          quantize_int8, save_whole_model)

fp32 = load_model()
int8 = quantize_int8(fp32)
print(int8)   # LSTM and Linear are now Dynamic Quantized modules
"""),
    md("""
## 1. Size and latency (fast, ~1 minute)
"""),
    code("""
rows = []
for name, m in [("baseline fp32", fp32), ("int8 dynamic", int8)]:
    rows.append({"model": name,
                 "size_MB": round(model_size_mb(m), 1),
                 "ms_per_prediction": round(benchmark_ms(m), 2)})
speed = pd.DataFrame(rows).set_index("model")
speed["speedup"] = (speed.loc["baseline fp32", "ms_per_prediction"]
                    / speed["ms_per_prediction"]).round(2)
speed
"""),
    md("""
## 2. Did accuracy survive? (~30 s per user per model on a laptop, ~5 min on Colab)
"""),
    code("""
acc = {}
for name, m in [("baseline fp32", fp32), ("int8 dynamic", int8)]:
    print(f"--- {name}")
    acc[name] = evaluate(m)
speed["mean_CLER"] = [acc[k].cler.mean() for k in speed.index]
speed["CLER_points_vs_baseline"] = (
    (speed.mean_CLER - speed.loc["baseline fp32", "mean_CLER"]) * 100).round(2)
speed
"""),
    code("""
save_whole_model(int8, "models/int8_dynamic.pt")
print("saved models/int8_dynamic.pt -- score it with:")
print("  python speed_report.py --model models/int8_dynamic.pt")
"""),
    md("""
## 3. Your turn: ideas to try

Each idea should end with `save_whole_model(m, "models/<name>.pt")`, then score it with
`speed_report.py`.

1. **Threads.** `benchmark_ms(m, threads=4)`. Faster, but the official rule is 1 thread. Why do you think that is?
2. **`torch.compile` / TorchScript.** `torch.jit.script(fp32)` or `torch.jit.trace`. Does the LSTM get faster?
3. **Prune the LSTM.** `torch.nn.utils.prune.l1_unstructured` on LSTM weights, then quantize.
4. **Smaller window = lower latency?** `benchmark_ms(m, window_s=0.25)`. What does a real-time system actually need?
5. **Half precision** (`m.half()`). It's usually *slower* on CPU. Measure it and explain why.
6. **Static quantization of the Conv1d** (harder). Dynamic quantization skipped the conv layer.

Note: retraining isn't feasible in one day on a CPU. Stick to post-training tricks.
"""),
])
