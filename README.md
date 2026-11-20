# Wearable AI Build Day — EMG gesture recognition

One-day build (Dec 4, 10 am–5 pm) by the Innovation Consulting Group. Four teams take
Meta's open wrist-EMG dataset and pretrained model, reproduce the baseline, make the
model smaller and faster, and ship a public web demo that recognizes hand gestures from
the muscle signals at the wrist.

> **License notice.** This project uses data, code and model weights from Meta's
> [generic-neuromotor-interface](https://github.com/facebookresearch/generic-neuromotor-interface),
> released under **CC BY-NC 4.0**. Everything here is for **non-commercial** use only,
> and the attribution below must stay with any copy. The short demo recordings
> bundled in `clips/` are excerpts of Meta's dataset, shared under the same license.
> Code we wrote for the day is released under the same terms, to keep things simple.

## The task

The wristband records **16 channels of surface EMG at 2 kHz**. The model, a conv layer
followed by a 3-layer LSTM, outputs a probability for each of 9 discrete gestures
200 times per second:

`index_press`, `index_release`, `middle_press`, `middle_release`, `thumb_click`,
`thumb_down`, `thumb_in`, `thumb_out`, `thumb_up`

A gesture "fires" when its probability crosses 0.35 (with 50 ms debounce).

**Metric: CLER (classification error rate).** Meta's metric for this task. Each
predicted gesture is matched to the nearest true one in time, and CLER is the share
that got the wrong label, averaged across gesture types. **Lower is better.** We
also show *gesture accuracy = 1 − CLER* because it reads better on a slide, but
comparisons use CLER.

**What CLER leaves out:** gestures the model misses entirely, and extra gestures
it invents. It only scores the label of each detected gesture. So "81.8%" means
"when it detects a gesture, it names it right 81.8% of the time", not "it catches
81.8% of gestures". On user_001's full session, 452 of 1,181 gestures were never
detected, while CLER was only 0.111. `match_events` (used in the Data notebook and
the app) shows misses and extras.

## Teams and files

| File | Team | What it does |
|---|---|---|
| `00_plot_emg.ipynb` | everyone | 10-minute warm-up: plot one recording and its gesture labels |
| `01_evaluate.ipynb` | Data | Run the pretrained model on the test users, reproduce the baseline, per-user chart |
| `02_speed.ipynb` | Speed | INT8 dynamic quantization, then compare accuracy, size and latency |
| `speed_report.py` | Speed / judges | One command that scores any model for the Speed challenge |
| `app.py` | Demo | Gradio app: pick a recording, see 16 channels, predicted vs. true gestures, latency |
| `emg_kit/` | shared | Loading, prediction, CLER, quantization and benchmarking, used by everything above |
| `make_clips.py` | organizers | Cuts the short demo recordings in `clips/` |

## On the day: one Colab notebook per team (recommended)

The easiest path. No install, no GitHub, no shared drive. Each notebook in `colab/` has
our code built into its first cell and downloads the model and 3 test users (1.1 GB)
straight into Colab in about a minute.

| Notebook | Team | Hands in |
|---|---|---|
| `colab/1_data_team.ipynb` | Data | `per_user_accuracy.png`, `findings.md` |
| `colab/2_speed_team.ipynb` | Speed | `demo_model.pt` |
| `colab/3_demo_team.ipynb` | Demo | `app.py` (runs live from Colab with a public link) |

**Organizers:** upload the three files to Google Drive and share each one as
*Anyone with the link → Viewer*. Students open the link, click **File → Save a copy in
Drive**, then **Runtime → Run all**. Rebuild the notebooks after changing `emg_kit`, `app.py` or the clips:

```bash
python tools/build_team_notebooks.py
jupyter nbconvert --to notebook --execute --output-dir /tmp/run colab/*.ipynb   # run once
python tools/save_outputs.py /tmp/run    # keep the outputs students should see
```

The saved outputs show students what to expect. Setup and Speed outputs are left
empty on purpose, because their timings depend on the machine.

"Run all" finishes in a few minutes in every notebook. The two long full-data runs (the
Data team's 3-user reproduction, about 16 min, and the Speed team's official score, about 30 min)
are switched off by default: set `RUN_FULL = True` / `RUN_OFFICIAL = True` to start them.
The Demo team's merge upload is behind `UPLOAD = True`, so it never stops "Run all".

Measured on free Colab (Nov 19, "Run all", three notebooks at once):

| Notebook | Setup | Whole notebook | Notes |
|---|---|---|---|
| Data | 138 s | **402 s (6.7 min)** | gesture check 1 min, quick 3-user check 2 min |
| Speed | 68 s | **581 s (9.7 min)** | 4 scoreboard rows, 1–2 min each |
| Demo | 77 s | **~90 s** | app launches with a public link |

Speed scoreboard on Colab (quick mode, 1 thread): baseline 0.0745 quick CLER at
156.8 ms; INT8 LSTM + Linear 0.0791 (+0.46 pts) at 68.2 ms (**2.3× faster**); INT8
LSTM only 0.0853 (+1.08) at 74.4 ms; INT8 Linear only 0.0745 at 152.2 ms.

Free Colab allows only a few sessions at once per Google account ("Too many sessions").
Each student uses their own account, so this only matters for organizers testing several notebooks.

The Speed team uses **quick mode** to try ideas: the first 5 minutes of each test user,
about 6× faster (`evaluate(..., minutes=5)` or `speed_report.py --quick`). Quick
numbers only compare with other quick numbers; judging uses the full test data.

## Setup (local, optional)

Python 3.12 or 3.13 (Colab works too: open a notebook, run the first cell). On Colab the install takes about 20 seconds because it keeps Colab's own PyTorch.

```bash
git clone https://github.com/sandeepvijayarao09/wearable-ai-build-day.git && cd wearable-ai-build-day
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m emg_kit.download --small-subset             # checkpoint (78 MB) + 3 users (1.1 GB)
```

Data goes to `~/emg_data` and the checkpoint to `~/emg_models`. Set `EMG_DATA_DIR` /
`EMG_MODELS_DIR` to put them somewhere else (e.g. a mounted shared drive).

**On the day, don't download the data yourself.** It's on the shared drive. Point
`EMG_DATA_DIR` at it. The full dataset is ~33 GB, and only the Data team needs all
10 test users.

Note: Meta's own notebooks load the checkpoint with Lightning, which breaks on
PyTorch ≥ 2.6 (`torch.load` now defaults to `weights_only=True`). `emg_kit.load_model`
handles this. If you run Meta's code directly, set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` and install `requirements-meta.txt` in a separate environment (Lightning 1.8 needs NumPy < 2).

Run the demo locally:

```bash
python app.py        # opens http://127.0.0.1:7860
```

## Baseline (Data team reproduces this)

Meta's pretrained checkpoint, evaluated with `emg_kit.evaluate` (equivalent to Meta's
Lightning `trainer.test`, which averages per-user CLER over the test users):

```bash
python speed_report.py --model baseline
```

| Test users | Mean CLER | Gesture accuracy |
|---|---|---|
| 3 users in small subset (000, 001, 002) | **0.1197** (0.1330 / 0.1106 / 0.1156) | 88.0% |
| **All 10 test users (full data): the official baseline** | **0.1819** | **81.8%** |

Per user (CLER): 000 0.133 · 001 0.111 · 002 0.116 · 003 0.243 · 004 0.113 ·
005 0.136 · 006 0.063 · 007 0.099 · 009 0.401 · 010 0.404. Two users (009, 010) are
much harder than the rest, which is a good story for the demo: the model works well
for most new people and badly for some.

Verified three ways on Nov 16 (PyTorch 2.8, MacBook CPU): `emg_kit.evaluate`, Meta's
Lightning `trainer.test` (0.18190), and Meta's own integration-test expectation
(`test_cler = 0.1819`, in `generic_neuromotor_interface/tests/expected_test_results.py`).
All 10 users take about 4.5 minutes on a laptop CPU.

Exact commands:

```bash
pip install -r requirements.txt
python -m emg_kit.download            # full data, ~33 GB (or point EMG_DATA_DIR at the shared drive)
python -c "from emg_kit.core import *; df = evaluate(load_model()); print(df, df.cler.mean())"
```

| Model (3 users, MacBook M-series, 1 thread) | Mean CLER | Size | ms per prediction |
|---|---|---|---|
| Baseline fp32 | 0.1197 | 25.9 MB | 12.5 |
| INT8 dynamic (LSTM + Linear) | 0.1275 (+0.78 pts) | 7.1 MB | 99.2 (8× **slower**) |

**Heads-up for the Speed team.** INT8 behaves in opposite ways on the two chip families
(measured Nov 16, 3 test users):

| Machine | fp32 ms | INT8 ms | INT8 CLER change |
|---|---|---|---|
| MacBook M5 Pro (`qnnpack`) | 12.5 | 99.2 (8× **slower**) | +0.78 pts |
| Colab CPU, 2 vCPU Xeon (`fbgemm`) | 144.9 | 60.5 (2.4× **faster**) | +0.49 pts |

INT8 accuracy also differs slightly by backend. So **every Speed entry is scored, for both
speed and CLER, on the same judging machine**.

## Will it run on my laptop?

Measured Nov 16:

| | MacBook M5 Pro | Colab CPU (2 vCPU Xeon, 12 GB) |
|---|---|---|
| Install `requirements.txt` | ~1 min (fresh) | **19 s** (keeps Colab's PyTorch) |
| Baseline, 3 users | 1.5 min | 16 min |
| Baseline, all 10 users | 4.2 min | ~55 min (estimated from the 3 users) |
| One prediction (1 s EMG, 1 thread) | 12.5 ms | 145 ms |
| Peak RAM | 2.8 GB | 2.0 GB |
| Demo app, one clip | ~0.15 s | ~1.5 s (estimated) |

Recordings run through the model in 60 s chunks with the LSTM state carried over
(identical results to one pass), so **8 GB laptops are fine**. Colab's CPU is about
**11× slower** than an M5 Pro, and extra threads don't help there. Plan for it:

- **Data team:** run the 10-user baseline on the fastest laptop in the room
  (minutes, not an hour), or reproduce the 3-user number (0.1197) on Colab.
- **Speed team:** iterate with `--users user_000` (one user, ~5 min on Colab), then
  score finalists on the judging machine.
- **Demo team:** Colab is fine. Clips are short.

## Speed challenge rule

> The fastest model, measured by median ms per prediction (1 s of EMG, batch 1,
> 1 CPU thread) with `speed_report.py`, wins, **as long as its mean CLER is no more
> than X points (X/100) above the baseline**. All models are scored on the judges' laptop.

X is set at the start of the day once the baseline is confirmed.

## Credits

**Technical lead:** Sandeep Vijayarao
**Students:** _added at 4 pm, only for people who agreed to be listed_

## Citation

If you use this work, cite Meta's paper and repo:

```bibtex
@article{generic_neuromotor_interface_2025,
  title   = {A generic non-invasive neuromotor interface for human-computer interaction},
  author  = {Kaifosh, Patrick and Reardon, Thomas R. and CTRL-labs at Reality Labs},
  journal = {Nature},
  year    = {2025},
  doi     = {10.1038/s41586-025-09255-w},
  issn    = {1476-4687},
  url     = {https://www.nature.com/articles/s41586-025-09255-w},
}
```

Code and data: https://github.com/facebookresearch/generic-neuromotor-interface (CC BY-NC 4.0).
