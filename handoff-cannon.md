# Handoff: cache [1.3.1] Linear Probes activations on Cannon

Written 2026-10-08 on makena, for a Claude Code session on Cannon. `CLAUDE.md` (auto-loaded) has the project context.

## Goal

Run the Llama models used in [1.3.1] Linear Probes on Cannon GPUs and save the activations, so Vadim
can do the probing exercises on makena (5 GB Pascal GPU, which cannot run 8B+ models). The script that
does this is already written and smoke-tested on makena with tiny stand-in models:
`chapter1_transformer_interp/exercises/part31_linear_probes/cache_activations.py`. Read its docstring.
Your job: set up the repo and environment on Cannon, get the models, run the script, verify the
outputs, and report. Do not change what the script computes: its data ordering, splits and tokenization
mirror `solutions.py` so that results on makena match the course. Fix real bugs only, commit any fix on
`vadim`, and push to `origin`.

## Steps

1. **Repo.** Clone the fork (branch `vadim`) to `~/p/ARENA_3.0`, then set remotes to match makena:
   ```bash
   git clone -b vadim https://github.com/vadsem/ARENA_materials.git ~/p/ARENA_3.0
   cd ~/p/ARENA_3.0 && git remote add upstream https://github.com/ARENA-education/ARENA_materials.git
   cd chapter1_transformer_interp/exercises
   git clone https://github.com/saprmarks/geometry-of-truth.git
   git clone https://github.com/ApolloResearch/deception-detection.git
   printf '/chapter1_transformer_interp/exercises/geometry-of-truth/\n/chapter1_transformer_interp/exercises/deception-detection/\n' >> ~/p/ARENA_3.0/.git/info/exclude
   ```
2. **HF token.** The script reads `HF_TOKEN` from `chapter1_transformer_interp/exercises/.env` (gitignored).
   It is not in git. Ask Vadim to copy it from makena (`scp` that file to the same path on Cannon); do
   not ask him to paste the token into the chat, and never print it. Then check gated access for all three models
   (`meta-llama/Llama-2-13b-hf`, `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.3-70B-Instruct`),
   e.g. with `huggingface_hub.model_info(name, token=...)` or `hf download <name> config.json`. If any
   is refused, stop and tell Vadim which ones he needs to request on huggingface.co.
3. **Environment.** Create conda env `arena` with Python 3.12 (do not reuse `cuda`, which LES-AI uses).
   Check the driver on a GPU node (`nvidia-smi`): with driver >= 580 the default PyPI torch (CUDA 13)
   is fine; otherwise install torch from the matching `https://download.pytorch.org/whl/cu12x` index.
   Then install the repo requirements with uv, keeping torch pinned and skipping `jax`, `brax`, `envpool`
   (not needed here). On makena this was:
   ```bash
   pip install uv
   pip freeze | grep -iE '^(torch|torchvision|torchaudio)==' > /tmp/constraints.txt
   grep -vE '^(jax|brax|envpool)' requirements.txt > /tmp/req.txt
   uv pip install --python $(which python) -c /tmp/constraints.txt -r /tmp/req.txt
   ```
   (Plain pip got stuck backtracking on this file; uv resolves it in seconds.)
4. **Model storage.** Set `HF_HOME` to scratch (e.g. `export HF_HOME=~/scratch/hf`). Weights total
   about 185 GB (13B: 26 GB, 8B: 16 GB, 70B: 140 GB bf16, quantized to int8 at load time). Download
   them before the GPU jobs, e.g. `hf download <name> --exclude "original/*"` in a CPU job, so GPU time
   is not spent downloading. Use whatever partition Cannon's conventions favour for that (`sinfo`).
5. **Smoke test** on `gpu_test` with the real models and `--limit 6`, writing to a throwaway directory:
   ```bash
   cd ~/p/ARENA_3.0/chapter1_transformer_interp/exercises/part31_linear_probes
   python cache_activations.py --out ~/scratch/arena/part31_smoke --limit 6 --stages got deception8b highstakes
   ```
   (Skip the 70B stage here: a MIG 3g.20gb slice cannot hold it.)
6. **Full runs**, as two sbatch jobs on the `gpu` partition (A100 80GB), output to
   `~/scratch/arena/part31_acts`. Run the script from the section folder; it finds the data relative
   to its own location.
   - Job 1: `--stages got deception8b highstakes`, 1 GPU, ~64 GB RAM (highstakes holds up to ~13 GB of
     activations in RAM while saving), 2 h limit should be plenty.
   - Job 2: `--stages deception70b`, 2 GPUs (int8 70B is ~70 GB, too tight for one 80 GB card with
     activations), ~128 GB RAM, 3 h limit (loading and quantizing 70B is the slow part).
7. **Verify** (load each file with `torch.load(path, weights_only=False)`) against the table below, and
   sanity-check the reference results: `got.pt["reference"]` p_diff_none should give >0.9 few-shot
   accuracy on sp_eval (sign of p_diff vs labels), `add` should raise and `subtract` lower mean p_diff;
   deception70b detector scores should be higher on deceptive than honest AI Liar dialogues on average
   (mean over `assistant_mask` tokens).
8. **Report** to Vadim: job IDs, wall times, file sizes, the sanity-check numbers, and the command for
   copying the cache to makena (run on makena):
   `rsync -avP cannon:scratch/arena/part31_acts/ ~/scratch/arena/part31_acts/`
   Never commit activations or `.env`.

## Expected outputs (`~/scratch/arena/part31_acts`)

| File | Contents | Approx. size |
|------|----------|--------------|
| `got.pt` | Llama-2-13b; per dataset (cities 1496, sp_en_trans 354, larger_than 1980, neg_cities 1496): `statements`, `labels`, `acts` [n, 40, 5120] bf16; `reference` few-shot / intervention p_diffs | 2.2 GB |
| `deception8b.pt` | Llama-3.1-8B-Instruct; `honest_acts`, `dishonest_acts` [512, 32, 4096]; `ai_liar` list of 54 dialogues with `acts` [seq, 32, 4096], `str_tokens`, `assistant_mask`; `reference` MM direction + steering generations | ~2 GB |
| `highstakes_train.pt` / `_test.pt` | layer-16 `acts` [3000 / 1000, 256, 4096] bf16, `masks`, `labels`, `texts`, `indices` | 6.3 / 2.1 GB |
| `highstakes_ood_anthropic_hh_balanced.pt` / `_toolace_balanced.pt` | same format, 2984 / 734 examples | 6.3 / 1.5 GB |
| `deception70b.pt` | Llama-3.3-70B int8; `ai_liar` with layer-22 `acts` [seq, 8192] fp16 and `detector_scores` [seq]; `detector` | ~0.2 GB |

Total about 20 GB.

## What the cache does not cover

These exercises need forward passes with Vadim's own code, so he will do them later in an interactive
GPU session on Cannon (not part of this handoff): the few-shot evaluation and causal interventions
(section 3), the deception steering hook (section 4), and running his own scoring code on 70B (section 5).
The script stores the solutions' results for these under `reference` so he can check his implementations.
