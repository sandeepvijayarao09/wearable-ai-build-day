"""Shared helpers for the Wearable AI Build Day.

Everything the notebooks, the speed script and the Gradio app need lives here,
so all four teams measure things the same way.

Metric: CLER (classification error rate), Meta's metric for discrete gestures.
Lower is better. CLER scores only the label of each DETECTED gesture: missed and
extra gestures are not counted (see match_events). We also show "gesture accuracy" = 1 - CLER because it is easier
to read on a slide, but the number we compare against is CLER.
"""

from __future__ import annotations

import io
import os
import statistics
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

warnings.filterwarnings("ignore", message=".*pkg_resources.*")

from generic_neuromotor_interface.cler import (  # noqa: E402
    DEBOUNCE,
    THRESHOLD,
    TOLERANCE,
    compute_cler,
    detect_gesture_events,
    get_matched_indices,
    map_gestures_to_probabilities,
    pad_dataframe,
)
from generic_neuromotor_interface.constants import (  # noqa: E402
    EMG_NUM_CHANNELS,
    EMG_SAMPLE_RATE,
    GestureType,
)
from generic_neuromotor_interface.networks import (  # noqa: E402
    DiscreteGesturesArchitecture,
)

TASK = "discrete_gestures"
GESTURES = [g.name for g in GestureType]  # 9 classes, index order = model output order

DATA_DIR = Path(os.environ.get("EMG_DATA_DIR", "~/emg_data")).expanduser()
MODELS_DIR = Path(os.environ.get("EMG_MODELS_DIR", "~/emg_models")).expanduser()
CKPT_PATH = MODELS_DIR / TASK / "model_checkpoint.ckpt"


# --------------------------------------------------------------------------- model

def load_model(path: str | Path | None = None) -> nn.Module:
    """Load Meta's pretrained discrete-gesture network as a plain torch module.

    Accepts either Meta's Lightning checkpoint (.ckpt) or a plain state_dict (.pt)
    written by `save_state_dict`. No Lightning or Hydra needed at inference time.
    """
    path = Path(path).expanduser() if path else CKPT_PATH
    _pick_quant_engine()  # quantized models need a backend set before unpickling
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, nn.Module):  # a whole pickled model (e.g. quantized)
        return blob.eval()
    state = blob.get("state_dict", blob)
    state = {k.removeprefix("network."): v for k, v in state.items()}
    model = DiscreteGesturesArchitecture()
    model.load_state_dict(state)
    return model.eval()


def save_state_dict(model: nn.Module, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def save_whole_model(model: nn.Module, path: str | Path) -> Path:
    """Quantized models can't be rebuilt from a float state_dict, so save the module."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, path)
    return path


def model_size_mb(model: nn.Module) -> float:
    """Size of the serialized model in MB (what you'd ship in the Space)."""
    buf = io.BytesIO()
    torch.save(model, buf)
    return buf.tell() / 1e6


def _pick_quant_engine() -> None:
    engines = torch.backends.quantized.supported_engines
    for e in ("fbgemm", "x86", "qnnpack"):
        if e in engines:
            torch.backends.quantized.engine = e
            return


def quantize_int8(model: nn.Module) -> nn.Module:
    """INT8 dynamic quantization of the LSTM and Linear layers (CPU only)."""
    _pick_quant_engine()
    return torch.ao.quantization.quantize_dynamic(
        model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
    ).eval()


@torch.no_grad()
def predict(model: nn.Module, emg: torch.Tensor | np.ndarray,
            times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run the model on one recording.

    emg:   (16, T) EMG at 2 kHz
    times: (T,) timestamps in seconds
    Returns probs (9, T') and their timestamps (T',). The network outputs at 200 Hz
    after a 20-sample left context (conv kernel 21, stride 10).
    """
    emg = torch.as_tensor(emg, dtype=torch.float32)
    if _streamable(model):
        probs = _forward_streaming(model, emg).numpy()
    else:  # e.g. a TorchScript or custom model: one big pass (needs more RAM)
        probs = torch.sigmoid(model(emg.unsqueeze(0)))[0].numpy()
    left, stride = 20, 10
    prob_times = np.asarray(times)[left::stride][: probs.shape[1]]
    return probs, prob_times


def _streamable(model: nn.Module) -> bool:
    return all(hasattr(model, a) for a in ("compression", "conv_layer", "relu",
                                            "post_conv_layer_norm", "lstm",
                                            "post_lstm_layer_norm", "projection"))


def _forward_streaming(model: nn.Module, emg: torch.Tensor,
                       chunk_steps: int = 12_000) -> torch.Tensor:
    """Same math as `model(emg)` but in 60 s chunks, carrying the LSTM state.

    A 42-minute recording in one pass peaks near 9 GB of RAM; this stays under
    ~1 GB. The conv layer gets its 20-sample left context from the previous
    chunk, and the LSTM's (h, c) is passed along, so results match the single
    pass to float precision (checked in tools/check_streaming.py).
    """
    k_total = (emg.shape[1] - 21) // 10 + 1
    out, state = [], None
    for k0 in range(0, k_total, chunk_steps):
        k1 = min(k0 + chunk_steps, k_total)
        x = emg[:, k0 * 10:(k1 - 1) * 10 + 21].unsqueeze(0)
        x = model.relu(model.conv_layer(model.compression(x)))
        x = model.post_conv_layer_norm(x.transpose(1, 2))
        x, state = model.lstm(x, state)
        x = model.projection(model.post_lstm_layer_norm(x))
        out.append(torch.sigmoid(x[0]).T)
    return torch.cat(out, dim=1)


def detect_events(probs: np.ndarray, prob_times: np.ndarray,
                  threshold: float = THRESHOLD, debounce: float = DEBOUNCE) -> pd.DataFrame:
    """Turn probabilities into discrete gesture events (Meta's exact rule)."""
    gp = map_gestures_to_probabilities(probs, prob_times)
    return detect_gesture_events(gp, threshold=threshold, debounce=debounce)


def cler(probs: np.ndarray, prob_times: np.ndarray, prompts: pd.DataFrame) -> float:
    return float(compute_cler(probs, prob_times, prompts))


def match_events(events: pd.DataFrame, prompts: pd.DataFrame,
                 tolerance: tuple[float, float] = TOLERANCE) -> pd.DataFrame:
    """Pair predicted events with true gestures exactly the way CLER does.

    Mirrors `generic_neuromotor_interface.cler.compute_confusion_matrix`: align in
    time, drop release events that don't match the finger currently pressed, align
    again. Returns one row per true gesture (matched or missed) plus one row per
    remaining unmatched prediction, with a `result` column.
    """
    labels = prompts[["name", "time"]].reset_index(drop=True).copy()
    labels["start"] = labels["time"]
    labels["end"] = labels["time"]
    events = events.reset_index(drop=True)
    if events.empty:
        return pd.DataFrame({"time": labels.time, "true": labels.name,
                             "predicted": "—", "result": "missed"})
    padded = pad_dataframe(labels, tolerance[0], tolerance[1])

    keep, state = [], "NEUTRAL"
    gt, pr = labels["name"].tolist(), events["name"].tolist()
    for p, g in get_matched_indices(events, padded):
        if p is not None:
            n = pr[p]
            if ((state == "NEUTRAL" and n not in ("index_release", "middle_release"))
                    or (state == "INDEX" and n != "middle_release")
                    or (state == "MIDDLE" and n != "index_release")):
                keep.append(p)
        if g is not None:
            state = {"index_press": "INDEX", "middle_press": "MIDDLE"}.get(gt[g], "NEUTRAL")
    kept = events.iloc[keep].reset_index(drop=True)

    rows = []
    for li, ri in get_matched_indices(padded, kept):
        if li is None:
            rows.append((kept.time[ri], "—", kept.name[ri], "extra"))
        elif ri is None:
            rows.append((labels.time[li], labels.name[li], "—", "missed"))
        else:
            ok = labels.name[li] == kept.name[ri]
            rows.append((labels.time[li], labels.name[li], kept.name[ri],
                         "correct" if ok else "wrong gesture"))
    df = pd.DataFrame(rows, columns=["time", "true", "predicted", "result"])
    return df.sort_values("time").reset_index(drop=True)


# ---------------------------------------------------------------------------- data

@dataclass
class Recording:
    user: str
    emg: np.ndarray       # (16, T) float32
    times: np.ndarray     # (T,) seconds (unix time)
    prompts: pd.DataFrame  # ground-truth gestures: columns name, time
    lead_in: float = 0.0   # seconds at the start used only to warm up the LSTM

    @property
    def seconds(self) -> float:
        return float(self.times[-1] - self.times[0])


def test_partitions(data_dir: Path = DATA_DIR) -> dict[str, tuple[Path, float, float]]:
    """Meta's 10 test users, pooled exactly like their eval (one span per user).

    Returns only users whose .hdf5 file is present, so this works with both the
    3-user small subset and the full dataset.
    """
    corpus = pd.read_csv(Path(data_dir) / f"{TASK}_corpus.csv")
    test = corpus[corpus["split"] == "test"]
    out = {}
    for ds, rows in test.groupby("dataset"):
        path = Path(data_dir) / ds
        if path.exists():
            user = f"user_{int(rows['user_number'].iloc[0]):03d}"
            out[user] = (path, float(rows["start"].min()), float(rows["end"].max()))
    return dict(sorted(out.items()))


def all_test_users(data_dir: Path = DATA_DIR) -> list[str]:
    corpus = pd.read_csv(Path(data_dir) / f"{TASK}_corpus.csv")
    t = corpus[corpus["split"] == "test"]
    return sorted(f"user_{int(u):03d}" for u in t["user_number"].unique())


def load_recording(path: Path, start: float = -np.inf, end: float = np.inf,
                   user: str = "") -> Recording:
    """Load one span of an HDF5 recording plus its gesture labels."""
    from generic_neuromotor_interface.data import EmgRecording

    with EmgRecording(Path(path), start, end) as rec:
        ts = rec[:]
        prompts = rec.prompts
    emg = np.ascontiguousarray(ts["emg"].T, dtype=np.float32)
    times = np.asarray(ts["time"], dtype=np.float64)
    prompts = prompts[prompts["name"].isin(GESTURES)]
    prompts = prompts[prompts["time"].between(times[0], times[-1])]
    return Recording(user or Path(path).stem, emg, times,
                     prompts[["name", "time"]].reset_index(drop=True))


def load_clip(path: str | Path) -> Recording:
    """Load a short .npz clip (what the demo ships with)."""
    z = np.load(path, allow_pickle=False)
    prompts = pd.DataFrame({"name": z["prompt_names"].astype(str),
                            "time": z["prompt_times"]})
    lead = float(z["lead_in"]) if "lead_in" in z else 0.0
    return Recording(str(z["user"]), z["emg"].astype(np.float32), z["times"], prompts, lead)


def save_clip(rec: Recording, t0: float, seconds: float, path: str | Path,
              lead_in: float = 3.0) -> Path:
    """Save [t0, t0+seconds) of a recording (t0 relative to its start) as .npz.

    `lead_in` extra seconds before t0 are included so the LSTM has warmed up by
    the time the shown part starts. Labels inside the lead-in are dropped.
    """
    a = rec.times[0] + t0
    i, j = np.searchsorted(rec.times, [a - lead_in, a + seconds])
    times = rec.times[i:j]
    p = rec.prompts[rec.prompts["time"].between(a, times[-1])]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, emg=rec.emg[:, i:j].astype(np.float16), times=times,
                        prompt_names=p["name"].to_numpy().astype("U32"),
                        prompt_times=p["time"].to_numpy(), user=np.array(rec.user),
                        lead_in=np.array(float(a - times[0])))
    return path


# ------------------------------------------------------------------------ evaluate

QUICK_MINUTES = 5.0  # practice mode: 5 minutes of each test user...
QUICK_WINDOWS = 5    # ...as 5 one-minute windows spread across the session


def evaluate(model: nn.Module, data_dir: Path = DATA_DIR, verbose: bool = True,
             users: list[str] | None = None,
             minutes: float | None = None) -> pd.DataFrame:
    """Per-user CLER on every test user available locally.

    Each user's 25-42 minute test span runs through the model as one sequence,
    exactly like Meta's Lightning test loop, which logs the mean of per-user CLER.
    So the headline number is `df.cler.mean()`.

    `minutes=5` (or `--quick` in speed_report) scores 5 one-minute windows spread
    evenly across each user's session, so every kind of gesture is included. It is
    ~6x faster and fine for comparing ideas, but it is NOT the official number:
    always compare quick to quick and full to full.
    """
    rows = []
    for user, (path, start, end) in test_partitions(data_dir).items():
        if users and user not in users:
            continue
        t = time.perf_counter()
        if minutes:
            rec, probs, pt = _predict_spread(model, path, start, end, user, minutes)
        else:
            rec = load_recording(path, start, end, user)
            probs, pt = predict(model, rec.emg, rec.times)
        secs = time.perf_counter() - t
        c = cler(probs, pt, rec.prompts)
        rows.append({"user": user, "cler": c, "accuracy": 1 - c,
                     "gestures": len(rec.prompts), "minutes": rec.seconds / 60,
                     "cpu_seconds": secs})
        if verbose:
            print(f"{user}: CLER {c:.4f}  ({len(rec.prompts)} gestures, "
                  f"{rec.seconds/60:.1f} min of EMG, {secs:.1f}s on CPU)")
    return pd.DataFrame(rows)


def _predict_spread(model, path, start, end, user, minutes, windows=QUICK_WINDOWS):
    """Predict on `windows` evenly spaced slices totalling `minutes` (quick mode).

    Each slice starts with a fresh LSTM state plus a few seconds of lead-in that are
    not scored. Slices are joined in time order, so CLER is computed exactly as usual.
    """
    w, lead = minutes * 60 / windows, 5.0
    recs, ps, pts = [], [], []
    for a in np.linspace(start + lead, end - w, windows):
        r = load_recording(path, a - lead, a + w, user)
        p, t = predict(model, r.emg, r.times)
        keep = t >= a
        recs.append(r.prompts[r.prompts.time >= a]); ps.append(p[:, keep]); pts.append(t[keep])
    prompts = pd.concat(recs, ignore_index=True)
    rec = Recording(user, np.empty((16, 0), np.float32), np.array([start, start + minutes * 60]), prompts)
    return rec, np.concatenate(ps, axis=1), np.concatenate(pts)


def predict_sample(model: nn.Module, user: str, minutes: float = 10,
                   data_dir: Path = DATA_DIR):
    """Predict on `minutes` of one test user, spread across the whole session.

    Faster than the full 25-42 min session but still covers every gesture type.
    Returns (rec, probs, prob_times); rec.prompts holds the true gestures.
    """
    path, start, end = test_partitions(data_dir)[user]
    return _predict_spread(model, path, start, end, user, minutes)


@torch.no_grad()
def benchmark_ms(model: nn.Module, window_s: float = 1.0, runs: int = 50,
                 warmup: int = 5, threads: int | None = 1) -> float:
    """Median milliseconds for one prediction on CPU.

    One prediction = one forward pass over `window_s` seconds of 16-channel EMG,
    batch size 1. We pin to 1 CPU thread by default so laptops are comparable.
    """
    old = torch.get_num_threads()
    if threads:
        torch.set_num_threads(threads)
    x = torch.randn(1, EMG_NUM_CHANNELS, int(window_s * EMG_SAMPLE_RATE))
    try:
        for _ in range(warmup):
            model(x)
        ts = []
        for _ in range(runs):
            t = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t) * 1000)
    finally:
        torch.set_num_threads(old)
    return statistics.median(ts)
