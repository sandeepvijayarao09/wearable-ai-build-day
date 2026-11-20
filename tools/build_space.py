"""Assemble a Hugging Face Space folder from this repo (organizers only).

    python tools/build_space.py            # -> ./space/
    # then: huggingface-cli upload <org>/<space-name> ./space . --repo-type=space

The Space runs on the free CPU tier and needs only the runtime subset of the
requirements (no Lightning, Hydra or Jupyter).
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys; sys.path.insert(0, str(ROOT))  # noqa: E402
OUT = ROOT / "space"

SPACE_README = """---
title: Wearable AI · Gestures from wrist EMG
emoji: ✋
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: {gradio}
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Recognize hand gestures from wrist EMG (Meta's open model)
---

Built at the Innovation Consulting Group Wearable AI Build Day (Dec 4).

Model, code and demo recordings: Meta,
[generic-neuromotor-interface](https://github.com/facebookresearch/generic-neuromotor-interface),
CC BY-NC 4.0. Kaifosh, Reardon & CTRL-labs at Reality Labs, *A generic non-invasive
neuromotor interface for human-computer interaction*, Nature (2025).
https://doi.org/10.1038/s41586-025-09255-w
"""

SPACE_REQS = """--extra-index-url https://download.pytorch.org/whl/cpu
generic_neuromotor_interface @ git+https://github.com/facebookresearch/generic-neuromotor-interface@b6bf250e2be5a67b23488104335373cdb87a15c9
torch>=2.4
numpy>=1.26
numba>=0.60
pandas>=2.2
omegaconf>=2.3
matplotlib>=3.8
"""


def main() -> None:
    import gradio
    import torch

    from emg_kit.core import load_model, save_state_dict

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    shutil.copy(ROOT / "app.py", OUT / "app.py")
    shutil.copytree(ROOT / "emg_kit", OUT / "emg_kit",
                    ignore=shutil.ignore_patterns("__pycache__", "download.py"))
    shutil.copytree(ROOT / "clips", OUT / "clips")
    if (ROOT / "assets").exists():
        shutil.copytree(ROOT / "assets", OUT / "assets")

    (OUT / "models").mkdir()
    demo = ROOT / "models" / "demo_model.pt"
    if demo.exists():  # Speed team's winner
        shutil.copy(demo, OUT / "models" / "demo_model.pt")
    else:
        save_state_dict(load_model(), OUT / "models" / "baseline_state.pt")

    (OUT / "README.md").write_text(SPACE_README.format(gradio=gradio.__version__))
    (OUT / "requirements.txt").write_text(SPACE_REQS)
    (OUT / ".gitattributes").write_text("*.pt filter=lfs diff=lfs merge=lfs -text\n"
                                        "*.npz filter=lfs diff=lfs merge=lfs -text\n")
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"Space folder ready: {OUT} ({size:.0f} MB, torch {torch.__version__})")


if __name__ == "__main__":
    main()
