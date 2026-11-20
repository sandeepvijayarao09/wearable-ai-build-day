# Demo clip credits

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
| `user_001_t1220.npz` | user_001 | 1220 | 10 | 17 |
| `user_002_t0470.npz` | user_002 | 470 | 10 | 27 |
| `user_000_t0520.npz` | user_000 | 520 | 10 | 27 |
| `user_001_t1310.npz` | user_001 | 1310 | 10 | 18 |
