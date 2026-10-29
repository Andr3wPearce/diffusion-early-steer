"""Probe-free early quality estimate: decode the model's own prediction of the final image at step i and score it
with ImageReward.

From data512/full trajectories: latents[i] = x before step i, latents[i+1] = x after it, and the FlowMatch Euler step
is x_{i+1} = x_i + (sigma_{i+1} - sigma_i) * v_i, so v_i (the CFG velocity) is exact and x0_hat_i = x_i - sigma_i * v_i.
No training, so every prompt (eval + train) is test data. Reports within-prompt Spearman rho vs final ImageReward,
top-1 hits and gain fraction per step. Saves previews for eval_0000 to data512/figures/xhat0_eval_0000.png.
"""
import json
from pathlib import Path

import run
import numpy as np
import torch
import ImageReward as RM
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from scipy.stats import spearmanr

DATA = run.HERE / "data512"
STEPS = [2, 5, 8, 13]  # t = (i+1)/28 = 0.11, 0.21, 0.32, 0.50


@torch.no_grad()
def main():
    device = "cuda"
    vae = AutoencoderKL.from_pretrained(run.MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16).to(device)
    vae.enable_slicing()  # decode one image at a time: same output, far less memory next to the generator
    proc = VaeImageProcessor(vae_scale_factor=8)
    ir_model = RM.load(str(run.IR_PT), device=device, med_config=str(run.IR_MED))
    files = sorted((DATA / "full").glob("eval_*.pt")) + sorted((DATA / "full").glob("train_*.pt"))
    results = {s: [] for s in STEPS}  # per step: list of (split, early scores, final IR)
    previews = {}
    for n, f in enumerate(files):
        d = torch.load(f, weights_only=False, mmap=True)
        lat, sig, final_ir = d["latents"], d["sigmas"].float(), d["ir"].numpy()
        for s in STEPS:
            x, x_next = lat[:, s].to(device).float(), lat[:, s + 1].to(device).float()
            v = (x_next - x) / (sig[s + 1] - sig[s])
            x0 = x - sig[s] * v
            img = vae.decode((x0 / vae.config.scaling_factor + vae.config.shift_factor).to(vae.dtype), return_dict=False)[0]
            pil = proc.postprocess(img, output_type="pil")
            early = np.array([ir_model.score(d["prompt"], p) for p in pil])
            results[s].append((d["split"], early, final_ir))
            if f.stem == "eval_0000":
                previews[s] = pil
        if (n + 1) % 5 == 0:
            print(f"{n + 1}/{len(files)} prompts", flush=True)

    def summarize(rows):
        rhos = [spearmanr(e, y).statistic for _, e, y in rows]
        hits = [int(e.argmax() == y.argmax()) for _, e, y in rows]
        pick = np.mean([y[e.argmax()] for _, e, y in rows]); rand = np.mean([y.mean() for _, e, y in rows])
        orc = np.mean([y.max() for _, e, y in rows])
        se = np.nanstd(rhos) / np.sqrt(len(rhos))
        return {"n": len(rows), "mean_rho": float(np.nanmean(rhos)), "rho_se": float(se), "top1_hits": int(sum(hits)),
                "top1_chance": len(rows) / 16, "gain_fraction": float((pick - rand) / (orc - rand))}

    out = {}
    print("\nstep  t     set     prompts  mean_rho (se)    top1 (chance)  gain")
    for s in STEPS:
        for name, rows in (("eval", [r for r in results[s] if r[0] == "eval"]), ("all", results[s])):
            m = summarize(rows); out[f"step{s}_{name}"] = m
            print(f"{s:4d}  {(s + 1) / 28:.2f}  {name:5s}  {m['n']:6d}   {m['mean_rho']:+.3f} ({m['rho_se']:.3f})   "
                  f"{m['top1_hits']:3d} ({m['top1_chance']:.1f})     {m['gain_fraction']:+.0%}")
    (DATA / "xhat0_baseline.json").write_text(json.dumps(out, indent=1))

    if previews:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        d = torch.load(DATA / "full" / "eval_0000.pt", weights_only=False, mmap=True)
        order = np.argsort(-d["ir"].numpy())[:6]
        fig, axes = plt.subplots(len(STEPS) + 1, len(order), figsize=(2.6 * len(order), 2.7 * (len(STEPS) + 1)))
        for j, k in enumerate(order):
            for i, s in enumerate(STEPS):
                axes[i, j].imshow(previews[s][k]); axes[i, j].axis("off")
                axes[i, j].set_title(f"x0_hat t={(s + 1) / 28:.2f}", fontsize=8)
            axes[-1, j].imshow(Image.open(DATA / "images" / f"eval_0000_seed{k}.png")); axes[-1, j].axis("off")
            axes[-1, j].set_title(f"final, seed {k}, IR {d['ir'][k]:+.2f}", fontsize=8)
        fig.suptitle(f'Model\'s predicted final image (x0_hat) during denoising: "{d["prompt"]}"', fontsize=11)
        fig.tight_layout(); fig.savefig(DATA / "figures" / "xhat0_eval_0000.png", dpi=100)


if __name__ == "__main__":
    main()
