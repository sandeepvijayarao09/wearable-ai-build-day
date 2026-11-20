"""Build the three self-contained team notebooks in colab/ (organizers only).

    python tools/build_team_notebooks.py

Each notebook carries our code (emg_kit, speed_report.py, demo clips) inside its
first cell, so students need no GitHub repo and no shared drive: open the link,
File -> Save a copy in Drive, Runtime -> Run all.

To save outputs into them (so students see results before running), execute them
once from a clean environment, e.g.:

    jupyter nbconvert --to notebook --execute --inplace colab/*.ipynb
"""

import base64
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "colab"
CORE = ["requirements.txt", "emg_kit/__init__.py", "emg_kit/core.py", "emg_kit/download.py"]

LICENSE = ("*Data, model and the code underneath: Meta, "
           "[generic-neuromotor-interface](https://github.com/facebookresearch/generic-neuromotor-interface), "
           "**CC BY-NC 4.0** (non-commercial, keep the credit). Kaifosh, Reardon & CTRL-labs at "
           "Reality Labs, Nature (2025).*")

HOW = """**How to use this notebook**
1. **File → Save a copy in Drive.** You now have your own copy. Edit anything.
2. **Runtime → Run all**, or run cells one by one with ▶ / Shift+Enter.
3. The outputs you see before running are from the organizers' run, so you know what to expect.
   On Colab things take longer than shown (about 10× for the big runs)."""


def blob(files):
    return {f: base64.b64encode((ROOT / f).read_bytes()).decode() for f in files}


def setup_cell(files, extra=""):
    return f'''# @title ⚙️ Setup: run this first (1–2½ min). Double-click to see the code.
import base64, os, subprocess, sys, time, warnings
warnings.filterwarnings("ignore")
IN_COLAB = "google.colab" in sys.modules
WORK = "/content/buildday" if IN_COLAB else os.path.abspath("buildday")
os.makedirs(WORK, exist_ok=True); os.chdir(WORK)
if WORK not in sys.path: sys.path.insert(0, WORK)
_t = time.time()
if "_T0" not in globals(): _T0 = _t   # start of the whole notebook run

FILES = {files!r}
for _f, _b in FILES.items():
    os.makedirs(os.path.dirname(_f) or ".", exist_ok=True)
    open(_f, "wb").write(base64.b64decode(_b))

def _run(args, what):
    r = subprocess.run([sys.executable, *args], capture_output=True, text=True)
    if r.returncode:
        print((r.stdout + r.stderr)[-2500:])
        raise RuntimeError(f"{{what}} failed. Show this cell's output to a helper.")

if not os.path.exists(".installed"):
    _run(["-m", "pip", "install", "-q", "-r", "requirements.txt"], "Install")
    open(".installed", "w").close()
_run(["-m", "emg_kit.download", "--small-subset"], "Download")  # model + 3 test users, 1.1 GB

from emg_kit.core import *
import pandas as pd, numpy as np, matplotlib.pyplot as plt
USERS = ["user_000", "user_001", "user_002"]  # the 3 test users in the download
os.makedirs("handin", exist_ok=True)
{extra}
print(f"✅ Ready in {{time.time() - _t:.0f}} s. Model and {{len(USERS)}} test users loaded.")'''


def download_cell(paths):
    return f'''# Download your hand-in files (they also stay in the handin/ folder on the left).
print(f"⏱ Whole notebook so far: {{time.time() - _T0:.0f}} s")
import shutil
files_to_send = {paths!r}
missing = [f for f in files_to_send if not os.path.exists(f)]
if missing:
    print("Not made yet:", missing)
elif IN_COLAB:
    from google.colab import files
    for f in files_to_send:
        files.download(f)
else:
    print("Saved:", files_to_send)'''


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s, clear=False):
    c = nbf.v4.new_code_cell(s.strip())
    if s.lstrip().startswith("# @title"):  # Colab shows only the title, code hidden
        c.metadata["cellView"] = "form"
    if clear:  # outputs that make no sense in a shared copy (local server URLs)
        c.metadata["tags"] = ["clear-output"]
    return c


def write(name, title, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    nb.metadata["colab"] = {"name": name, "provenance": []}
    OUT.mkdir(exist_ok=True)
    path = OUT / name
    nbf.write(nb, path)
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.1f} MB) · {title}")


BASELINE_CSV = (ROOT / "results" / "baseline_10users.csv").read_text()

# ========================================================================== DATA
write("1_data_team.ipynb", "Data team", [
    md(f"""
# 📊 Data team · How well does the model read new people?

Meta trained a model that turns wrist muscle signals (EMG) into 9 finger and thumb
gestures. Your job: **find out how well it works on people it has never seen, and
explain why it sometimes fails.**

**You hand in** (at the merge):
- `per_user_accuracy.png`: the chart that goes into the live demo
- `findings.md`: 3 short findings for the final slide

**The metric is CLER** (lower is better): of the gestures the model *detects*, the share
it names wrong, averaged over the 9 gesture types. *Gesture accuracy* = 1 − CLER.
**Careful:** CLER doesn't count gestures the model **misses** completely, or extra ones it
invents. Step 3 shows those, and they're a good finding.

{HOW}

{LICENSE}
"""),
    code(setup_cell(blob(CORE))),

    md("""
## 1 · Look at the signal (5 min)
One minute of one test user. Each grey line is one of the 16 electrodes around the
wrist; each coloured line is a moment the person made a gesture. Try another user or
another minute by changing the first two lines.
"""),
    code("""
user = "user_000"     # try user_001, user_002
minute = 3            # which minute of their session to show

path, start, end = test_partitions()[user]
rec = load_recording(path, start + minute * 60, start + minute * 60 + 10, user)  # 10 seconds
t = rec.times - rec.times[0]
colors = dict(zip(GESTURES, plt.cm.tab10.colors))
fig, ax = plt.subplots(figsize=(13, 6))
gap = 4 * rec.emg.std()
for ch in range(16):
    ax.plot(t, rec.emg[ch] + ch * gap, lw=0.5, color="0.45")
for _, g in rec.prompts.iterrows():
    x = g.time - rec.times[0]
    ax.axvline(x, color=colors[g["name"]], lw=1.5)
    ax.text(x, 16 * gap, g["name"].replace("_", " "), rotation=90, fontsize=8,
            ha="right", va="bottom", color=colors[g["name"]])
ax.set_yticks([]); ax.set_xlabel("seconds"); ax.spines[["top", "right", "left"]].set_visible(False)
ax.set_title(f"{user}: 16 EMG channels, 10 seconds, {len(rec.prompts)} gestures", loc="left")
plt.show()
"""),

    md("""
## 2 · The official baseline (already computed)
Meta's test set has **10 people** the model never saw in training. The organizers ran
the model on all 10 (that takes about an hour on Colab, so it's done for you). Your
download has only 3 of them, which you'll reproduce in step 4.

This cell makes the chart that goes into the demo. Make it yours: colours, title,
labels, sorting. Then re-run it.
"""),
    code(f'''
import io
BASELINE = pd.read_csv(io.StringIO("""{BASELINE_CSV}"""))
mean = BASELINE.cler.mean()
print(f"Mean CLER over 10 test users: {{mean:.4f}}  ->  gesture accuracy {{1 - mean:.1%}}")

b = BASELINE.sort_values("accuracy")
fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.barh(b.user.str.replace("user_", "person "), b.accuracy * 100, color="#5b6cff")
ax.axvline((1 - mean) * 100, color="k", ls="--", lw=1)
ax.text((1 - mean) * 100, -0.9, f" average {{(1 - mean):.0%}}", fontsize=9)
ax.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=9)
ax.set_xlim(0, 105); ax.set_xlabel("detected gestures named correctly (%), = 1 − CLER")
ax.set_title("How well the model reads 10 people it never saw", loc="left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("handin/per_user_accuracy.png", dpi=150)
plt.show()
BASELINE.round(3)
'''),

    md("""
## 3 · Which gestures does it get wrong? (about 1 min on Colab)
Runs the model on 10 minutes of one person (sampled across their whole session, so
every gesture type shows up) and lines up every true gesture with what the model
predicted, exactly the way CLER counts them.
"""),
    code("""
user = "user_001"     # try user_000, user_002
model = load_model()
rec, probs, pt = predict_sample(model, user, minutes=10)
matches = match_events(detect_events(probs, pt), rec.prompts)

print(f"{user}, 10 sampled minutes: CLER {cler(probs, pt, rec.prompts):.3f}")
print(matches.result.value_counts().to_string(), "\\n")
per_gesture = (matches[matches.true != "—"]
               .assign(correct=lambda d: d.result == "correct")
               .groupby("true").correct.mean().sort_values())
per_gesture.mul(100).round(1).rename("% correct").to_frame()
"""),
    code("""
# Confusion: rows = what the person did, columns = what the model said ("—" = nothing)
pd.crosstab(matches.true, matches.predicted)
"""),

    md("""
## 4 · Reproduce the baseline on your 3 users
By default this is a **quick check** (5 sampled minutes per person, about 2 min on
Colab), which should give **0.0745**. For the real reproduction, set `RUN_FULL = True`
and run it again. It takes about 16 min on Colab and should give **0.1197**
(0.1330 / 0.1106 / 0.1156). Work on your findings while it runs.
"""),
    code("""
RUN_FULL = False   # <- True for the full 3-user reproduction (~16 min on Colab)

df = evaluate(load_model(), users=USERS, minutes=None if RUN_FULL else QUICK_MINUTES)
expected = 0.1197 if RUN_FULL else 0.0745
print(f"\\nMean CLER: {df.cler.mean():.4f}  (expected {expected}, "
      f"{'full sessions' if RUN_FULL else 'quick check'})")
df.to_csv("handin/baseline_3users.csv", index=False)
df.round(4)
"""),

    md("""
## 5 · Your findings (for the final slide)
Write 3 short findings in the cell below, then run it. Ideas to look into:
- Two people (009 and 010) are far worse than the rest. What could make someone harder
  to read? (Arm size, how they wear the band, how they make the gesture…)
- Which gestures are easy and which get confused with each other (step 3)?
- How many gestures does the model miss entirely? CLER ignores those. Does the headline
  number tell the whole story?
- Is 82% good enough for a product? What would a user feel at that error rate?
"""),
    code('''
findings = """
1. ...
2. ...
3. ...
"""
open("handin/findings.md", "w").write(findings.strip() + "\\n")
print(findings)
'''),
    md("## 6 · Hand in"),
    code(download_cell(["handin/per_user_accuracy.png", "handin/findings.md"])),
])

# ========================================================================= SPEED
write("2_speed_team.ipynb", "Speed team", [
    md(f"""
# ⚡ Speed team · Make the model faster without making it worse

**The challenge:** the fastest model wins, **as long as its CLER is no more than X points
above the baseline** (X is announced at the start of the day).

- **Speed** = median milliseconds for one prediction (1 second of EMG, 1 CPU thread).
- **CLER** = of the gestures the model detects, the share it names wrong (lower is better).

**You hand in** `demo_model.pt`, your best model. The judges score every entry on the
same machine with `speed_report.py`, so your numbers here are practice.

**Use quick mode while you try things.** `score()` below uses 5 one-minute windows spread
across each test user's session, 1–2 min per idea on Colab instead of 16. Quick numbers only
compare with other quick numbers, and they are noisier: if two ideas are within a few
tenths of a point, the official run decides.

**Heads-up:** INT8 is about 2.4× *faster* on Colab (Intel) but 8× *slower* on Apple
laptops. Always measure on the machine you're judged on.

{HOW}

{LICENSE}
"""),
    code(setup_cell(blob(CORE + ["speed_report.py"]))),

    md("""
## 1 · Your scoreboard
`score(model, "name")` adds a row: quick CLER, Δ points vs. the baseline, size, speed.
The first row is the baseline.
"""),
    code("""
BOARD, MODELS = [], {}

def score(model, name):
    df = evaluate(model, users=USERS, minutes=QUICK_MINUTES, verbose=False)
    BOARD.append({"model": name, "quick CLER": round(df.cler.mean(), 4),
                  "MB": round(model_size_mb(model), 1), "ms": round(benchmark_ms(model), 1)})
    MODELS[name] = model
    b = pd.DataFrame(BOARD)
    b["Δ pts"] = ((b["quick CLER"] - b["quick CLER"][0]) * 100).round(2)
    b["× faster"] = (b.ms[0] / b.ms).round(2)
    return b

baseline = load_model()
score(baseline, "baseline (fp32)")
"""),

    md("""
## 2 · First idea: INT8 quantization
Store the LSTM and output layer weights as 8-bit integers instead of 32-bit floats:
4× smaller, and on Intel CPUs, faster.
"""),
    code("""
score(quantize_int8(baseline), "int8 LSTM + Linear")
"""),

    md("""
## 3 · Your ideas
Copy the pattern: build a model, then `score()` it. Some starting points:
- Quantize only part of the model (cells below).
- Look at `print(baseline)`: where does the time go? The conv layer, the 3 LSTM layers?
- Bigger ideas (smaller LSTM, pruning, distillation) need retraining on the full
  dataset with a GPU, which won't fit in a day. Pitch them in your slide instead.
"""),
    code("""
import torch
from torch import nn
lstm_only = torch.ao.quantization.quantize_dynamic(baseline, {nn.LSTM}, dtype=torch.qint8).eval()
score(lstm_only, "int8 LSTM only")
"""),
    code("""
linear_only = torch.ao.quantization.quantize_dynamic(baseline, {nn.Linear}, dtype=torch.qint8).eval()
score(linear_only, "int8 Linear only")
"""),
    code("""
# Your idea here
"""),

    md("""
## 4 · Hand in your best model
Put the name of your best row in `BEST`. Then, if you have 30 minutes left, set
`RUN_OFFICIAL = True` in the next cell for the official check (full test data, the same
script the judges use).
"""),
    code("""
BEST = "int8 LSTM + Linear"        # <- the name from your scoreboard
save_whole_model(MODELS[BEST], "handin/demo_model.pt")
print(f"Saved {BEST} -> handin/demo_model.pt ({os.path.getsize('handin/demo_model.pt') / 1e6:.1f} MB)")
"""),
    code("""
RUN_OFFICIAL = False   # <- True for the official score (~30 min on Colab)
X = 2                  # <- the number announced in the morning

if RUN_OFFICIAL:
    !{sys.executable} speed_report.py --model handin/demo_model.pt --x {X}
else:
    print("Skipped. Set RUN_OFFICIAL = True to run the official score.")
""", clear=True),
    code(download_cell(["handin/demo_model.pt"])),
])

# ========================================================================== DEMO
# Demo clips are cut from the data the setup cell downloads anyway, instead of
# embedding 4.6 MB of EMG in the notebook (which makes Colab's editor crawl).
CLIPS = [("user_001", 1220), ("user_002", 470), ("user_000", 520), ("user_001", 1310)]
CUT_CLIPS = f"""
for _u, _t0 in {CLIPS!r}:  # 10 s demo clips (Meta's data, see clips/CREDITS.md)
    _out = f"clips/{{_u}}_t{{_t0:04d}}.npz"
    if not os.path.exists(_out):
        _p, _s, _e = test_partitions()[_u]
        save_clip(load_recording(_p, _s + _t0 - 5, _s + _t0 + 15, _u), 5, 10, _out)"""
write("3_demo_team.ipynb", "Demo team", [
    md(f"""
# ✋ Demo team · A live demo anyone can try

You run the web app that shows the model reading gestures from real wrist recordings.
It's already working. Your job: **make it clear, good-looking, and tell the story**, then
add the other teams' work at the merge.

**You hand in** `app.py`, which goes live as the public demo.

{HOW}

{LICENSE}
"""),
    code(setup_cell(blob(CORE + ["clips/CREDITS.md"]), CUT_CLIPS)),

    md("""
## 1 · The app's code
This cell **is** `app.py`. Edit it right here, run it (it saves the file), then run the
launch cell below. Everything worth changing is marked `TODO`.
"""),
    code("%%writefile app.py\n" + (ROOT / "app.py").read_text()),

    md("""
## 2 · Launch it
Gives you a public link (`…gradio.live`, works for 72 hours) you can open on your phone
or send to your team. Re-run this cell after every change to the cell above.

**Don't press ■ Stop on a cell while the app is running.** It also cuts the public link.
If the link shows "No interface is running", just re-run this cell for a new one.
"""),
    code("""
import importlib, gradio as gr
gr.close_all()
import app; importlib.reload(app)
app.demo.launch(share=IN_COLAB, quiet=True)
""", clear=True),

    md("""
## 3 · Ideas
- Write the intro for someone who has never heard of EMG. Who is it for, why does it matter?
- Team names and one sentence about what each team found.
- Make the plot easier to read: fewer channels, bigger labels, your colours.
- A "what does each gesture look like" section (photos or emoji).

## 4 · At the merge (1:15): add the other teams' work
Set `UPLOAD = True` below, run the cell, and pick the Speed team's `demo_model.pt` and the
Data team's `per_user_accuracy.png`.
The app picks them up automatically. Then re-run the launch cell.
"""),
    code("""
UPLOAD = False   # <- set to True at the merge, then run this cell and pick the files

if UPLOAD and IN_COLAB:
    from google.colab import files
    for name, data in files.upload().items():
        if name.endswith(".pt"):
            dest = "models/demo_model.pt"
        elif name.endswith(".png"):
            dest = "assets/per_user_accuracy.png"
        else:
            print("skipped", name, "(expected a .pt model or a .png chart)"); continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        open(dest, "wb").write(data)
        print("added", dest)
""", clear=True),
    md("## 5 · Hand in"),
    code(download_cell(["app.py"])),
])
