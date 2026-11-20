"""Download Meta's discrete-gestures data and pretrained checkpoint.

Same URLs and behaviour as Meta's `generic_neuromotor_interface/scripts/download_*.py`.
We re-implement the two entry points here because Meta's `scripts/` folder has no
`__init__.py`, so it is not included when the package is pip-installed from git.

Usage:
    python -m emg_kit.download --small-subset          # 3 users, ~1.1 GB
    python -m emg_kit.download                          # full data, ~33 GB (organizers only)
    python -m emg_kit.download --models-only
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from generic_neuromotor_interface.download_utils import (
    download_file,
    ensure_dir,
    extract_tar,
)

BASE_URL = "https://fb-ctrl-oss.s3.amazonaws.com/neuromotor-data"
TASK = "discrete_gestures"


def download_data(emg_data_dir: str | Path, dataset: str = "small_subset",
                  task: str = TASK, cleanup: bool = True) -> Path:
    data_dir = ensure_dir(Path(emg_data_dir).expanduser())
    flag = data_dir / f".data_downloaded_{dataset}_{task}"
    if flag.exists():
        when = datetime.fromtimestamp(flag.stat().st_mtime)
        print(f"{task} {dataset} already in {data_dir} (downloaded {when}).")
        return data_dir

    tar_path = data_dir / f"{task}_{dataset}.tar"
    download_file(f"{BASE_URL}/data/{task}/{dataset}.tar", tar_path,
                  f"Downloading {dataset} data for {task}")
    extract_tar(tar_path, data_dir, "Unpacking")
    if cleanup:
        tar_path.unlink()

    corpus = data_dir / f"{task}_corpus.csv"
    download_file(f"{BASE_URL}/data/{task}/{task}_corpus.csv", corpus,
                  "Downloading corpus CSV (train/val/test split)")
    flag.touch()
    print(f"Done: {data_dir}")
    return data_dir


def download_models(model_dir: str | Path, task: str = TASK) -> Path:
    task_dir = ensure_dir((Path(model_dir) / task).expanduser())
    if (task_dir / "model_checkpoint.ckpt").exists():
        print(f"Checkpoint already in {task_dir}.")
        return task_dir
    tar_path = task_dir / f"{task}.tar"
    download_file(f"{BASE_URL}/checkpoints/{task}/{task}.tar", tar_path,
                  f"Downloading {task} checkpoint")
    extract_tar(tar_path, task_dir, "Unpacking")
    tar_path.unlink()
    print(f"Done: {task_dir}")
    return task_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=os.environ.get("EMG_DATA_DIR", "~/emg_data"))
    p.add_argument("--models-dir", default=os.environ.get("EMG_MODELS_DIR", "~/emg_models"))
    p.add_argument("--small-subset", action="store_true",
                   help="3 users (~1.1 GB) instead of the full ~33 GB dataset")
    p.add_argument("--models-only", action="store_true")
    a = p.parse_args()

    download_models(a.models_dir)
    if not a.models_only:
        download_data(a.data_dir, "small_subset" if a.small_subset else "full_data")


if __name__ == "__main__":
    main()
