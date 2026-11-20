"""Copy outputs from executed team notebooks into colab/ (organizers only).

    python tools/save_outputs.py <folder with executed notebooks>

Keeps the outputs students should see before running, and clears the ones that
would mislead them:
- the setup cell (its timing reflects the organizer's network, not Colab's)
- cells tagged "clear-output" (local server links, upload prompts, long runs)
- every output of the Speed notebook, because latency depends on the machine
  (an Apple laptop makes INT8 look 5x slower; on Colab it is 2.4x faster)
"""

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
MACHINE_SPECIFIC = {"2_speed_team.ipynb"}


def main(src_dir: str) -> None:
    for src in sorted(Path(src_dir).glob("*_team.ipynb")):
        nb = nbformat.read(src, as_version=4)
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        for i, c in enumerate(code_cells):
            if (i == 0 or src.name in MACHINE_SPECIFIC
                    or "clear-output" in c.metadata.get("tags", [])):
                c.outputs, c.execution_count = [], None
        text = nbformat.writes(nb)
        assert "/Users/" not in text and "/private/" not in text, f"local path in {src.name}"
        (ROOT / "colab" / src.name).write_text(text)
        print("saved", src.name)


if __name__ == "__main__":
    main(sys.argv[1])
