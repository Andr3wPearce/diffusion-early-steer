"""Replicate ProbeSelect's early quality prediction on SD 3.5 Medium.

4 seeds of one prompt. At the first step i with (i+1)/28 >= 0.2 the released SD3-M probe
(workdir/SD3M_IR_ema.ckpt) scores each candidate; the noisy latent at that step is decoded as a
preview. Final images are scored with ImageReward. Mirrors ProbeSelect_SD3.__call__ (src/new_pipe/SD3.py)
at commit db757424ccd2fceae5a19939020e7ce62c7f1791, one candidate per pipeline call.
"""
import gc
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
# External code and weights (see README "Setup"); override with environment variables.
REPO = Path(os.environ.get("PROBESELECT_DIR", HERE / "third_party" / "ProbeSelect"))  # Guhuary/ProbeSelect @ db75742
CKPT = REPO / "workdir" / "SD3M_IR_ema.ckpt"
IMAGEREWARD_DIR = Path(os.environ.get("IMAGEREWARD_DIR", HERE / "weights" / "ImageReward"))
IR_PT = IMAGEREWARD_DIR / "ImageReward.pt"
IR_MED = IMAGEREWARD_DIR / "med_config.json"
COMPBENCH_DIR = Path(os.environ.get("COMPBENCH_DIR", HERE / "third_party" / "T2I-CompBench" / "examples" / "dataset"))
OUT = HERE / "outputs"

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
sys.path.insert(0, str(REPO))

import torch
import ImageReward as RM
from diffusers import StableDiffusion3Pipeline
from scipy.stats import spearmanr

from src.models.components.model import ProbHead
from src.data_gen.utils import ActivationGrabber_SD3

# stabilityai/stable-diffusion-3.5-medium (gated): a Hub id or a local diffusers-layout folder.
MODEL_ID = os.environ.get("SD35M_PATH", "stabilityai/stable-diffusion-3.5-medium")
PROMPT = "A bowl of fruit on a table in an apartment."
SEEDS = [0, 1, 2, 3]
STEPS = 28
GUIDANCE = 7.0
SIZE = 1024
CKPT_MD5 = "2f7b3c84a3dae02e59e0c54f8f698ea3"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_probe(device):
    # Arguments exactly as in ProbeSelect_SD3.from_pretrained.
    probe = ProbHead(
        latent_channels=48, latent_height=48, num_layers=1,
        latent_width=48, hidden_dims=[128, 256], hidden_dims_mlp=[512], use_batch_norm=True,
        activation="relu", dropout_rate=0.5,
        resnet_eps=1e-5, resnet_act_fn="silu",
        resnet_groups=16, attention_head_dim=5,
        output_scale_factor=1.0, downsample_type="conv",
        temb_channels=512, resnet_time_scale_shift="default",
        num_channels_time=64, text_dim=768, has_text_embed=True,
    )
    probe.load_state_dict(torch.load(CKPT, weights_only=True))
    return probe.eval().float().to(device)


@torch.no_grad()
def blip_text_embedding(ir_model, prompt, device):
    # Same call as ProbeSelect_SD3.__call__: BLIP tokenizer + text_encoder in 'textonly' mode, CLS token.
    text_input = ir_model.blip.tokenizer([prompt], padding="max_length", truncation=True, max_length=35, return_tensors="pt").to(device)
    return ir_model.blip.text_encoder(
        text_input.input_ids, attention_mask=text_input.attention_mask,
        encoder_hidden_states=None, encoder_attention_mask=None, return_dict=True, mode="textonly",
    ).last_hidden_state[:, 0, :].float()  # [1, 768]


@torch.no_grad()
def decode(pipe, latents):
    latents = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents.to(pipe.vae.dtype), return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    torch.cuda.empty_cache()
    return image


def make_figure(results, rho, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    n = len(results)
    fig = plt.figure(figsize=(4 * n + 5, 8.5))
    gs = fig.add_gridspec(2, n + 1, width_ratios=[1] * n + [1.25], wspace=0.08, hspace=0.18)
    for k, r in enumerate(results):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(Image.open(OUT / r["xt_preview"])); ax.axis("off")
        ax.set_title(f"seed {r['seed']}  x_t @ t={r['t_normalized']:.3f}\nprobe {r['probe_score']:.4f} (rank {r['probe_rank']})", fontsize=11)
        ax = fig.add_subplot(gs[1, k])
        ax.imshow(Image.open(OUT / r["final_image"])); ax.axis("off")
        ax.set_title(f"final  ImageReward {r['imagereward']:+.3f} (rank {r['imagereward_rank']})", fontsize=11)
    ax = fig.add_subplot(gs[:, n])
    xs, ys = [r["probe_score"] for r in results], [r["imagereward"] for r in results]
    ax.scatter(xs, ys, s=60)
    for r in results:
        ax.annotate(f"seed {r['seed']}", (r["probe_score"], r["imagereward"]), textcoords="offset points", xytext=(6, 4), fontsize=10)
    ax.set_xlabel("probe score at ~20% (relative, 0-1)")
    ax.set_ylabel("ImageReward (final image)")
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_title(f"Spearman rho = {rho:+.3f}  (n={n})")
    ax.grid(alpha=0.3)
    fig.suptitle(f'ProbeSelect SD3-M probe vs ImageReward: "{PROMPT}"', fontsize=13)
    fig.savefig(path, dpi=110, bbox_inches="tight")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    assert md5(CKPT) == CKPT_MD5, "probe checkpoint md5 mismatch"

    # 1. Encode the prompt with the three text encoders on CPU, then free them (16 GB GPU).
    t0 = time.perf_counter()
    te_pipe = StableDiffusion3Pipeline.from_pretrained(MODEL_ID, transformer=None, vae=None, torch_dtype=torch.bfloat16)
    pe, npe, ppe, nppe = te_pipe.encode_prompt(PROMPT, None, None, device="cpu", do_classifier_free_guidance=True)
    del te_pipe
    gc.collect()
    print(f"prompt encoded on CPU in {time.perf_counter() - t0:.0f}s")

    # 2. Transformer + VAE on GPU, prompt embeddings passed in.
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_ID, text_encoder=None, text_encoder_2=None, text_encoder_3=None,
        tokenizer=None, tokenizer_2=None, tokenizer_3=None, torch_dtype=torch.bfloat16,
    ).to(device)
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    embeds = dict(prompt_embeds=pe.to(device), negative_prompt_embeds=npe.to(device),
                  pooled_prompt_embeds=ppe.to(device), negative_pooled_prompt_embeds=nppe.to(device))

    probe = build_probe(device)
    ir_model = RM.load(str(IR_PT), device=device, med_config=str(IR_MED))
    text_emb = blip_text_embedding(ir_model, PROMPT, device)
    grabber = ActivationGrabber_SD3(pipe.transformer)

    results = []
    for seed in SEEDS:
        rec = {}

        def on_step(_pipe, i, timestep, kwargs):
            if "probe_score" not in rec and (i + 1) / STEPS >= 0.2:
                feats = grabber.get_pca_feature().to(device).float()  # [1, 48, 48, 48]
                t_norm = torch.full((feats.shape[0],), (i + 1) / STEPS, device=device)
                with torch.no_grad():
                    score, _ = probe(feats, t_norm, text_emb=text_emb)
                rec.update(probe_score=float(score), step_index=i, t_normalized=(i + 1) / STEPS, timestep=float(timestep))
                decode(_pipe, kwargs["latents"]).save(OUT / f"xt_seed{seed}.png")
            grabber.latest_feats.clear()
            return {}

        t0 = time.perf_counter()
        image = pipe(
            **embeds, num_inference_steps=STEPS, guidance_scale=GUIDANCE, height=SIZE, width=SIZE,
            generator=torch.Generator("cpu").manual_seed(seed),
            callback_on_step_end=on_step, callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]
        torch.cuda.empty_cache()
        image.save(OUT / f"final_seed{seed}.png")
        results.append({"seed": seed, **rec, "xt_preview": f"xt_seed{seed}.png", "final_image": f"final_seed{seed}.png"})
        print(f"seed {seed}: probe {rec['probe_score']:.5f} at step {rec['step_index']} (t={rec['t_normalized']:.3f}), {time.perf_counter() - t0:.1f}s")

    grabber.close()
    for r in results:
        r["imagereward"] = float(ir_model.score(PROMPT, str(OUT / r["final_image"])))
    for key, rank_key in (("probe_score", "probe_rank"), ("imagereward", "imagereward_rank")):
        for rank, r in enumerate(sorted(results, key=lambda r: -r[key]), 1):
            r[rank_key] = rank
    rho = float(spearmanr([r["probe_score"] for r in results], [r["imagereward"] for r in results]).statistic)

    summary = {
        "prompt": PROMPT, "seeds": SEEDS, "steps": STEPS, "guidance_scale": GUIDANCE, "size": SIZE,
        "probe_checkpoint": "ProbeSelect/workdir/SD3M_IR_ema.ckpt", "probe_checkpoint_md5": CKPT_MD5,
        "probeselect_commit": "db757424ccd2fceae5a19939020e7ce62c7f1791",
        "spearman_rho": rho,
        "probe_top1_seed": min(results, key=lambda r: r["probe_rank"])["seed"],
        "imagereward_top1_seed": min(results, key=lambda r: r["imagereward_rank"])["seed"],
        "results": results,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0), "platform": platform.platform(),
            "notes": "prompt encoded on CPU; VAE tiling enabled; probe run in float32",
        },
    }
    (OUT / "results.json").write_text(json.dumps(summary, indent=2))
    make_figure(results, rho, OUT / "figure.png")
    for r in results:
        print(f"seed {r['seed']}: probe {r['probe_score']:.5f} (rank {r['probe_rank']})  ImageReward {r['imagereward']:+.4f} (rank {r['imagereward_rank']})")
    print(f"Spearman rho = {rho:+.3f}; probe picks seed {summary['probe_top1_seed']}, ImageReward best is seed {summary['imagereward_top1_seed']}")


if __name__ == "__main__":
    if "--figure-only" in sys.argv:  # re-render the figure from an existing results.json
        s = json.loads((OUT / "results.json").read_text())
        make_figure(s["results"], s["spearman_rho"], OUT / "figure.png")
    else:
        main()
