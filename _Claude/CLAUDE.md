# ARENA course (Vadim's working copy)

Vadim is working through the ARENA curriculum (https://learn.arena.education) in this repo. Upstream is
`ARENA-education/ARENA_materials`; Vadim's fork is `vadsem/ARENA_materials` (public). Work happens on
the `vadim` branch; merge `upstream/main` into it to pick up course updates.

Remotes: `origin` = fork (push here), `upstream` = ARENA-education/ARENA_materials (read only).

## Workflow

- Do each section in a new `answers.py` in the section folder (e.g.
  `chapter1_transformer_interp/exercises/part31_linear_probes/answers.py`), using `# %%` cells, copying
  setup code from the course page. Never edit the upstream exercise/solution files.
- The setup code finds the repo root by walking up from the working directory, so run cells with the
  section folder as cwd (VS Code: `"jupyter.notebookFileRoot": "${fileDirname}"`, set in `.vscode/`).
- `chapter1_transformer_interp/exercises/.env` holds `HF_TOKEN` (gitignored; never commit or print it).
- `geometry-of-truth/` and `deception-detection/` in `chapter1_transformer_interp/exercises/` are
  separate clones needed by [1.3.1]; excluded via `.git/info/exclude`, not committed.

## Machines

- **makena** (local): Quadro P2200, 5 GB, Pascal sm_61. Conda env `arena` (Python 3.10, cloned from
  `cuda`): torch 2.14.1+cu126 from the PyTorch cu126 index (CUDA 12.8+ wheels dropped Pascal). Keep torch
  pinned when installing anything (`-c constraints` or `uv pip install` with a torch pin). ARENA
  `requirements.txt` installed except jax, brax, envpool (need Python 3.12 / unsupported on Pascal).
  Fine for small models (GPT-2, ~1B); anything larger runs on Cannon.
- **Cannon** (FASRC, `ssh cannon`): A100 GPUs. Used to run the Llama models and cache activations;
  see `handoff-cannon.md`. Large activation caches live in scratch, never in the repo or Dropbox:
  makena `~/scratch/arena/`, Cannon `~/scratch/arena/`.

## Status

- [1.3.1] Linear Probes: setup works on makena. Activations for the whole section are cached on Cannon
  by `part31_linear_probes/cache_activations.py` (see `handoff-cannon.md` for what it stores and which
  exercises still need the live model).
