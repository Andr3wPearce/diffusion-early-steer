# diffusion-early-steer

Can we tell early in denoising which Stable Diffusion 3.5 Medium samples will turn out well, and can we steer them?

We started by replicating [ProbeSelect](https://github.com/Guhuary/ProbeSelect), a probe that predicts final image quality
at ~20% of denoising. We then built our own predictors, compared them against a training-free baseline (ImageReward on
the model's own x̂₀ prediction), and tested steering by re-noising candidates mid-trajectory.

**Results and figures: [report.md](report.md).**

## Setup

Tested on Windows 11 with an RTX 5080 (16 GB) and Python 3.12. The scripts are written to fit in 16 GB of VRAM and
32 GB of RAM.

### 1. Python environment

```bash
python3.12 -m venv .venv
.venv/Scripts/python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install "setuptools<70" wheel
.venv/Scripts/python -m pip install --no-build-isolation image-reward==1.5 openai-clip==1.0.1 fairscale==0.4.13 timm==0.6.13
.venv/Scripts/python -m pip install -r requirements.txt
```

### 2. External code, weights and prompts

The default locations are listed below. Each one can be overridden with the environment variable shown.

| What | Default location | Env var | How to get it |
|---|---|---|---|
| ProbeSelect repo (for `ProbHead`, `ActivationGrabber_SD3` and the released checkpoint) | `third_party/ProbeSelect` | `PROBESELECT_DIR` | `git clone https://github.com/Guhuary/ProbeSelect third_party/ProbeSelect && git -C third_party/ProbeSelect checkout db757424ccd2fceae5a19939020e7ce62c7f1791` (the checkpoint `workdir/SD3M_IR_ema.ckpt` should have md5 `2f7b3c84a3dae02e59e0c54f8f698ea3`) |
| ImageReward weights | `weights/ImageReward/{ImageReward.pt, med_config.json}` | `IMAGEREWARD_DIR` | `hf download zai-org/ImageReward ImageReward.pt med_config.json --local-dir weights/ImageReward` |
| T2I-CompBench++ prompts | `third_party/T2I-CompBench/examples/dataset` | `COMPBENCH_DIR` | `git clone --depth 1 --filter=blob:none --sparse https://github.com/Karine-Huang/T2I-CompBench third_party/T2I-CompBench && git -C third_party/T2I-CompBench sparse-checkout set examples/dataset` |
| SD 3.5 Medium (gated: accept the license on Hugging Face, then run `hf auth login`) | Hub id `stabilityai/stable-diffusion-3.5-medium` | `SD35M_PATH` | Loads from the Hub cache by default. To use a local copy, run `hf download stabilityai/stable-diffusion-3.5-medium --exclude "*.fp16*" --exclude "sd3.5_medium.safetensors" --exclude "text_encoders/*" --local-dir weights/sd35m` and set `SD35M_PATH=weights/sd35m`. If downloads stall, set `HF_HUB_DISABLE_XET=1`. |

## Scripts

Every script imports `run.py` for paths and generation settings: 28 steps, guidance 7.0, bf16, and CPU-seeded noise.
Outputs are written next to the scripts, in git-ignored folders.

| Stage | Script | What it does | Output |
|---|---|---|---|
| **Replicate ProbeSelect** | `run.py` | Runs the released SD3-M probe on 1 prompt × 4 seeds at 1024px, and scores the final images with ImageReward. | `outputs/` |
| | `run_batch.py` | Same as `run.py`, for 20 prompts × 8 seeds. We ran the first 10. | `outputs_20x8/` |
| | `diagnose.py` | Re-scores those candidates under variants: timestep, CFG row, time label, and a random-feature sensitivity check. | `outputs_20x8/diagnostics.json` |
| **Data** | `gen_train.py` | 1024px training data, 8 seeds per prompt. Superseded by `gen512.py`. | `train_data/` |
| | `gen512.py` | Open-ended 512px dataset with 16 seeds per prompt. Saves the whole trajectory: block-20 features for both CFG rows at all 28 steps, the initial noise, x_t after every step, the final images and ImageReward scores. Resumable. | `data512/` |
| | `extract_steps.py` | Caches a few timesteps from `data512/full` for quick experiments. | `data512/steps/` |
| **Probes** | `train_probe.py` | Trains ProbeSelect's `ProbHead` on correctly paired features, using a within-prompt ranking loss. | `runs/` |
| | `simple_probe.py` | Ridge ranker on sign-invariant summaries of the PCA features. | stdout |
| **x̂₀ baseline** | `xhat0_baseline.py` | Decodes x̂₀ = x_t − σ·v at several steps and scores it with ImageReward. No training. | `data512/xhat0_baseline.json` |
| | `xhat0_features.py` | Per-candidate x̂₀ ImageReward score and its 768-d embedding at 8 steps. | `data512/xhat0/` |
| | `combo_probe.py` | Early-selection curves: x̂₀ score vs probes vs combinations, at each step. | `data512/early_selection.json`, figure |
| **Steering** | `steer.py` | Keeps the top 4 at t≈0.29, branches each into 4 with fresh noise, selects at t=0.5, then finishes. Its sampler reproduces the pipeline bit for bit. | `data512/steer/` |
| | `explore_noise.py` | Visual sweep of noise amounts (η) and rewind depths on one candidate per prompt. | figures |
| | `rewind.py` | Judges at t=0.5 and rewinds the bottom 12 of 16 to steps 8…0 with fresh noise. Every rewound candidate is finished and scored. | `data512/rewind/` |
| | `rewind_draws.py` | 16 noise draws per depth on a single weak candidate. Written but not yet run. | figures |
| **Utilities** | `plot_ir.py`, `plot_rewind_prompt.py` | ImageReward summary figure; per-prompt rewind grid. | `data512/figures/` |
| | `status.py`, `status_loop.sh` | Self-refreshing local HTML status page for long runs. | `data512/status.html` |

A typical order: `run.py` → `run_batch.py` → `diagnose.py`, then `gen512.py` (left running), then `xhat0_features.py`
→ `combo_probe.py` → `steer.py` / `rewind.py`.

## Gotchas we hit

- **ProbeSelect's feature grabber takes `output[1][0::2]`.** That assumes the CFG batch is interleaved. Diffusers
  actually orders it `[uncond × B, cond × B]`, so with batch size > 1 the features land on the wrong candidates. See
  report §2. Our scripts generate one candidate per call when replicating, and slice rows explicitly when batching.
- **`StableDiffusion3Pipeline.encode_prompt` is not under `no_grad`.** If you keep many embeddings, each one holds
  T5's autograd graph (about 4.4 GB). On the CPU this showed up as random access-violation crashes.
- **Matching the scheduler exactly in a custom Euler loop.** `(sigma_next - sigma)` is a 0-dim fp32 tensor, so its
  product with the bf16 velocity stays bf16 before the fp32 add. Mirroring that makes the loop reproduce the pipeline
  bit for bit (`steer.py --validate`).
- **Hugging Face xet downloads can hang silently,** and a hung download was restarted from zero. Use
  `HF_HUB_DISABLE_XET=1` or a resumable `curl -C -`.
- **Keep the machine from sleeping** during long runs.
