"""Data for training our own probe (ProbeSelect architecture, correctly paired features).

Same generation settings and feature recipe as run.py / diagnose.py (verified to reproduce the released-probe
inputs exactly), one candidate per pipeline call, so every feature belongs to its own image. At step 5
(t = 6/28) both CFG rows of transformer block 20 are kept (uncond = what [0::2] gives at batch 1; cond).

Phase A (eval): the 80 candidates of outputs_20x8/p00-p09 (the released probe's test set), rerun to completion
  (final images checked against outputs_20x8); ImageReward labels from outputs_20x8/results.json.
Phase B (train): 200 new prompts (25 per category from T2I-CompBench *_train.txt, disjoint from the val
  prompts used for eval) x 8 seeds, generated to completion and scored.
Outputs (nothing generated is thrown away except the pre-PCA block-20 activations, ~1 TB):
  train_data/{eval/pNN, train/NNN}.pt   compact step-5 data for train_probe.py: feat_uncond / feat_cond [8, 48, 48, 48]
                                        fp16, ir [8], text_emb [768], prompt, category, seeds
  train_data/full/{eval_pNN, train_NNN}.pt  whole trajectories: features of both CFG rows at all 28 steps, initial
                                        noise + x_t after every step, sigma/timestep schedule (see save_full)
  train_data/train_images/              final images
Resumable: finished files are skipped.
"""
import json
import os
import random
import subprocess
import sys
import time

import run
import numpy as np
import torch
import ImageReward as RM
from diffusers import StableDiffusion3Pipeline
from PIL import Image

from diagnose import RowGrabber

DATA = run.HERE / "train_data"
EVAL_SRC = run.HERE / "outputs_20x8"
SEEDS = list(range(8))
STEP = 5  # first i with (i+1)/28 >= 0.2
TRAIN_PER_CATEGORY = 25
CATEGORIES = ["color", "shape", "texture", "spatial", "3d_spatial", "non_spatial", "numeracy", "complex"]
COMPBENCH = run.COMPBENCH_DIR


def train_prompts():
    rng = random.Random(1)
    rows = []
    for cat in CATEGORIES:
        lines = [l.strip() for l in (COMPBENCH / f"{cat}_train.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
        rows += [{"category": cat, "prompt": p} for p in rng.sample(lines, TRAIN_PER_CATEGORY)]
    return rows


def encode_only():
    """Child process: CPU bf16 text encoding (as in run.py), saved to disk."""
    prompts = [p["prompt"] for p in train_prompts()]
    t0 = time.perf_counter()
    te = StableDiffusion3Pipeline.from_pretrained(run.MODEL_ID, transformer=None, vae=None, torch_dtype=torch.bfloat16)
    with torch.no_grad():  # encode_prompt is not under no_grad; without it every embedding keeps T5's graph
        embeds = [te.encode_prompt(p, None, None, device="cpu", do_classifier_free_guidance=True) for p in prompts]
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = DATA / "train_prompt_embeds.pt.tmp"
    torch.save({"prompts": prompts, "embeds": [tuple(t.contiguous() for t in e) for e in embeds]}, tmp)
    tmp.replace(DATA / "train_prompt_embeds.pt")
    print(f"{len(prompts)} train prompts encoded on CPU in {time.perf_counter() - t0:.0f}s", flush=True)
    os._exit(0)


def run_candidates(pipe, grabber, embeds, seeds):
    """Run each seed to completion (one candidate per call), keeping the whole trajectory:
    block-20 PCA features of both CFG rows at every step, the initial noise and x_t after every step."""
    pe, npe, ppe, nppe = (t.to("cuda") for t in embeds)
    noise = {}

    def grab_noise(module, args, kwargs):  # first transformer call of a run sees the initial noise
        if "x" not in noise:
            noise["x"] = kwargs["hidden_states"][0:1].detach().cpu()

    pre = pipe.transformer.register_forward_pre_hook(grab_noise, with_kwargs=True)
    out = []
    try:
        for seed in seeds:
            noise.clear()
            uncond, cond, latents = [], [], []

            def on_step(_pipe, i, timestep, kwargs):
                uncond.append(grabber.features(0).half().cpu())
                cond.append(grabber.features(1).half().cpu())
                latents.append(kwargs["latents"].detach().cpu())
                return {}

            image = pipe(prompt_embeds=pe, negative_prompt_embeds=npe, pooled_prompt_embeds=ppe, negative_pooled_prompt_embeds=nppe,
                         num_inference_steps=run.STEPS, guidance_scale=run.GUIDANCE, height=run.SIZE, width=run.SIZE,
                         generator=torch.Generator("cpu").manual_seed(seed),
                         callback_on_step_end=on_step, callback_on_step_end_tensor_inputs=["latents"]).images[0]
            torch.cuda.empty_cache()
            out.append({
                "image": image,
                "feat_uncond": torch.cat(uncond),             # [28, 48, 48, 48] fp16, step i = feature of forward pass i
                "feat_cond": torch.cat(cond),                 # [28, 48, 48, 48] fp16
                "latents": torch.cat([noise["x"]] + latents), # [29, 16, 128, 128] bf16: initial noise, then x after step i
            })
    finally:
        pre.remove()
    return out


def save_full(path, prompt, category, cands, ir, text_emb, scheduler):
    tmp = path.with_suffix(".tmp")
    torch.save({
        "feat_uncond": torch.stack([c["feat_uncond"] for c in cands]),  # [8, 28, 48, 48, 48]
        "feat_cond": torch.stack([c["feat_cond"] for c in cands]),      # [8, 28, 48, 48, 48]
        "latents": torch.stack([c["latents"] for c in cands]),          # [8, 29, 16, 128, 128]
        "sigmas": scheduler.sigmas.cpu(), "timesteps": scheduler.timesteps.cpu(),
        "ir": torch.tensor(ir), "text_emb": text_emb.squeeze(0).cpu(), "prompt": prompt, "category": category,
        "seeds": SEEDS, "steps": run.STEPS, "guidance_scale": run.GUIDANCE, "size": run.SIZE,
        "notes": "feat_*[k, i]: block-20 features (ActivationGrabber_SD3 recipe, PCA to 48) from forward pass i of seed k; "
                 "uncond = CFG row 0 (what [0::2] picks at batch 1), cond = row 1. latents[k, 0] = initial noise, "
                 "latents[k, i+1] = x after step i. CFG velocity at step i = (latents[i+1]-latents[i])/(sigmas[i+1]-sigmas[i]).",
    }, tmp)
    tmp.replace(path)


def save_group(path, prompt, category, uncond, cond, ir, text_emb):
    tmp = path.with_suffix(".tmp")
    torch.save({"feat_uncond": torch.cat(uncond), "feat_cond": torch.cat(cond), "ir": torch.tensor(ir),
                "text_emb": text_emb.squeeze(0).cpu(), "prompt": prompt, "category": category, "seeds": SEEDS,
                "step": STEP, "t": (STEP + 1) / run.STEPS}, tmp)
    tmp.replace(path)


def main():
    assert run.md5(run.CKPT) == run.CKPT_MD5
    (DATA / "eval").mkdir(parents=True, exist_ok=True)
    (DATA / "train").mkdir(parents=True, exist_ok=True)
    (DATA / "train_images").mkdir(parents=True, exist_ok=True)
    prompts = train_prompts()
    (DATA / "train_prompts.json").write_text(json.dumps(prompts, indent=1))

    emb_path = DATA / "train_prompt_embeds.pt"
    for attempt in range(1, 4):
        if emb_path.exists():
            break
        rc = subprocess.run([sys.executable, "-u", __file__, "--encode-only"]).returncode
        print(f"encode subprocess attempt {attempt} exited {rc}", flush=True)
    train_embeds = torch.load(emb_path, weights_only=True)
    assert train_embeds["prompts"] == [p["prompt"] for p in prompts]

    pipe = StableDiffusion3Pipeline.from_pretrained(
        run.MODEL_ID, text_encoder=None, text_encoder_2=None, text_encoder_3=None,
        tokenizer=None, tokenizer_2=None, tokenizer_3=None, torch_dtype=torch.bfloat16,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    ir_model = RM.load(str(run.IR_PT), device="cuda", med_config=str(run.IR_MED))
    grabber = RowGrabber(pipe.transformer)

    (DATA / "full").mkdir(parents=True, exist_ok=True)

    # Phase A: eval candidates (released probe's test set), full trajectories. Final images must match outputs_20x8.
    batch = json.loads((EVAL_SRC / "results.json").read_text())
    eval_embeds = torch.load(EVAL_SRC / "prompt_embeds.pt", weights_only=True)
    for e in batch["prompts"]:
        full_path = DATA / "full" / f"eval_p{e['index']:02d}.pt"
        if full_path.exists():
            continue
        cands = run_candidates(pipe, grabber, eval_embeds["embeds"][e["index"]], SEEDS)
        same = all(np.array_equal(np.asarray(c["image"]), np.asarray(Image.open(EVAL_SRC / f"p{e['index']:02d}" / f"final_seed{s}.png")))
                   for s, c in zip(SEEDS, cands))
        ir = [next(r["imagereward"] for r in e["results"] if r["seed"] == s) for s in SEEDS]
        text_emb = run.blip_text_embedding(ir_model, e["prompt"], "cuda")
        save_group(DATA / "eval" / f"p{e['index']:02d}.pt", e["prompt"], e["category"],
                   [c["feat_uncond"][STEP:STEP + 1] for c in cands], [c["feat_cond"][STEP:STEP + 1] for c in cands], ir, text_emb)
        save_full(full_path, e["prompt"], e["category"], cands, ir, text_emb, pipe.scheduler)
        print(f"eval p{e['index']:02d} full trajectories saved; final images identical to outputs_20x8: {same}", flush=True)

    # Phase B: training prompts, generated to completion and scored with ImageReward.
    for idx, p in enumerate(prompts):
        path = DATA / "train" / f"{idx:03d}.pt"
        full_path = DATA / "full" / f"train_{idx:03d}.pt"
        if path.exists() and full_path.exists():
            continue
        t0 = time.perf_counter()
        cands = run_candidates(pipe, grabber, train_embeds["embeds"][idx], SEEDS)
        ir = []
        for s, c in zip(SEEDS, cands):
            img_path = DATA / "train_images" / f"{idx:03d}_seed{s}.png"
            c["image"].save(img_path)
            ir.append(float(ir_model.score(p["prompt"], str(img_path))))
        text_emb = run.blip_text_embedding(ir_model, p["prompt"], "cuda")
        save_group(path, p["prompt"], p["category"],
                   [c["feat_uncond"][STEP:STEP + 1] for c in cands], [c["feat_cond"][STEP:STEP + 1] for c in cands], ir, text_emb)
        save_full(full_path, p["prompt"], p["category"], cands, ir, text_emb, pipe.scheduler)
        print(f"train {idx:03d}/{len(prompts)} [{p['category']}] IR spread {max(ir) - min(ir):.2f} ({time.perf_counter() - t0:.0f}s)", flush=True)

    grabber.handle.remove()
    print("ALL DATA DONE", flush=True)


if __name__ == "__main__":
    encode_only() if "--encode-only" in sys.argv else main()
