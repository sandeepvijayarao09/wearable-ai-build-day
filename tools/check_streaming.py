"""Check that chunked (streaming) prediction matches one full pass.

    python tools/check_streaming.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from emg_kit.core import (_forward_streaming, cler, load_model, load_recording,  # noqa: E402
                          quantize_int8, test_partitions)

path, s, e = test_partitions()["user_002"]
rec = load_recording(path, s, s + 300, "user_002")   # 5 minutes
emg = torch.as_tensor(rec.emg)
times = rec.times[20::10]

for name, m in [("fp32", load_model()), ("int8", quantize_int8(load_model()))]:
    with torch.no_grad():
        full = torch.sigmoid(m(emg.unsqueeze(0)))[0].numpy()
        stream = _forward_streaming(m, emg, chunk_steps=7_000).numpy()  # forces 9 chunks
    print(f"{name}: max |diff| = {np.abs(full - stream).max():.2e}  "
          f"CLER full {cler(full, times, rec.prompts):.6f}  "
          f"streaming {cler(stream, times, rec.prompts):.6f}")
