"""Open-ended probe training data at 512x512, 16 candidates per prompt (SD 3.5 Medium).

Same settings and feature recipe as gen_train.py except size 512 and 16 seeds: 28 steps, guidance 7.0,
bf16, CPU-seeded noise, one candidate per pipeline call, block-20 features (ActivationGrabber_SD3 recipe,
PCA to 48) of both CFG rows at every step, initial noise + x_t after every step, ImageReward of the final image.

Eval first: run.py's prompt + 3 per category from T2I-CompBench *_val.txt (25 prompts).
Then train prompts: all of *_train.txt, shuffled (seed 0) and interleaved across the 8 categories so any prefix
is category-balanced. Runs until stopped; resumable (finished prompts are skipped).

Prompt encoding (CPU bf16, no_grad) runs in a background child process in chunks, ahead of generation.

Layout under data512/:
  prompts.json                     ordered prompt list (eval entries first)
  embeds/chunk_NNN.pt              text embeddings, CHUNK prompts per file
  compact/{eval|train}_NNNN.pt     step-5 (t = 6/28) features, ir [16], text_emb [768]  (for train_probe.py)
  full/{eval|train}_NNNN.pt        whole trajectories: feat_uncond/feat_cond [16, 28, 48, 48, 48] fp16,
                                   latents [16, 29, 16, 64, 64] bf16, sigmas, timesteps, ir, text_emb, ...
  images/{eval|train}_NNNN_seedK.png
"""
import json
import os
import random
import subprocess
import sys
import time
from itertools import zip_longest

import run
import torch
import torch.nn.functional as F
import ImageReward as RM
from diffusers import StableDiffusion3Pipeline

from diagnose import RowGrabber
from src.data_gen.utils import pca_channel_per_sample

DATA = run.HERE / "data512"
SIZE = 512
SEEDS = list(range(16))
STEP = 5  # first i with (i+1)/28 >= 0.2
CHUNK = 50
LOOKAHEAD = 2
BATCH = 16  # candidates per pipeline call
CATEGORIES = ["color", "shape", "texture", "spatial", "3d_spatial", "non_spatial", "numeracy", "complex"]
COMPBENCH = run.COMPBENCH_DIR


def build_prompt_list():
    read = lambda name: [l.strip() for l in (COMPBENCH / name).read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(0)
    evals = [{"split": "eval", "category": "coco-style (run.py)", "prompt": run.PROMPT}]
    for cat in CATEGORIES:
        evals += [{"split": "eval", "category": cat, "prompt": p} for p in rng.sample(read(f"{cat}_val.txt"), 3)]
    eval_set = {e["prompt"] for e in evals}
    per_cat = []
    for cat in CATEGORIES:
        lines = list(dict.fromkeys(p for p in read(f"{cat}_train.txt") if p not in eval_set))
        rng.shuffle(lines)
        per_cat.append([{"split": "train", "category": cat, "prompt": p} for p in lines])
    seen, train = set(), []
    for row in zip_longest(*per_cat):
        for item in row:
            if item and item["prompt"] not in seen:
                seen.add(item["prompt"]); train.append(item)
    rows = evals + train
    for i, r in enumerate(rows):
        r["id"] = f"{r['split']}_{i:04d}"
    return rows


def encode_worker():
    """Child process: encode the missing chunks among first_chunk .. first_chunk + LOOKAHEAD on CPU (as in run.py),
    then exit so the ~11 GB of text encoders is released between bursts. Started by main() at below-normal
    priority; 4 threads still encode far faster than generation consumes prompts."""
    torch.set_num_threads(4)
    first = int(sys.argv[sys.argv.index("--encode-worker") + 1])
    rows = json.loads((DATA / "prompts.json").read_text())
    todo = [c for c in range(first, first + LOOKAHEAD + 1)
            if c * CHUNK < len(rows) and not (DATA / "embeds" / f"chunk_{c:03d}.pt").exists()]
    if not todo:
        os._exit(0)
    te = StableDiffusion3Pipeline.from_pretrained(run.MODEL_ID, transformer=None, vae=None, torch_dtype=torch.bfloat16)
    for ci in todo:
        c, path = ci * CHUNK, DATA / "embeds" / f"chunk_{ci:03d}.pt"
        prompts = [r["prompt"] for r in rows[c:c + CHUNK]]
        with torch.no_grad():  # encode_prompt is not under no_grad; without it every embedding keeps T5's graph
            embeds = [tuple(t.contiguous() for t in te.encode_prompt(p, None, None, device="cpu", do_classifier_free_guidance=True))
                      for p in prompts]
        tmp = path.with_suffix(".tmp")
        torch.save({"prompts": prompts, "embeds": embeds}, tmp)
        tmp.replace(path)
    os._exit(0)


def get_embeds(i, rows, cache):
    c = i // CHUNK
    path = DATA / "embeds" / f"chunk_{c:03d}.pt"
    if c not in cache:
        while not path.exists():
            time.sleep(5)
        cache.clear()
        cache[c] = torch.load(path, weights_only=True)
        assert cache[c]["prompts"][0] == rows[c * CHUNK]["prompt"]
    return cache[c]["embeds"][i % CHUNK]


class BatchRowGrabber(RowGrabber):
    """RowGrabber for a batch of B candidates. diffusers orders the CFG batch [uncond x B, cond x B], so the
    candidate-k rows are k (uncond) and B + k (cond) -- not interleaved, which is what [0::2] wrongly assumes."""

    def features_rows(self, rows):
        x = self.latest[rows]
        bs, n, c = x.shape
        p, new_c = self.patch, c // self.patch // self.patch
        h = w = int(n ** 0.5)
        x = x.view(bs, h, w, p, p, new_c).permute(0, 5, 1, 3, 2, 4).reshape(bs, new_c, h * p, w * p)
        x = F.interpolate(x, size=(48, 48), mode="bilinear", align_corners=False)
        return pca_channel_per_sample(x, low_c=48)  # per-sample PCA, same as one at a time


def run_candidates(pipe, grabber, embeds, seeds, batch_size):
    """Generate the candidates batch_size at a time. Each candidate keeps its own CPU-seeded initial noise."""
    pe, npe, ppe, nppe = (t.to("cuda") for t in embeds)
    out = []
    for start in range(0, len(seeds), batch_size):
        chunk = seeds[start:start + batch_size]
        B = len(chunk)
        noise, uncond, cond, latents = {}, [], [], []

        def grab_noise(module, args, kwargs):  # first transformer call sees the initial noise (uncond half)
            if "x" not in noise:
                noise["x"] = kwargs["hidden_states"][:B].detach().cpu()

        def on_step(_pipe, i, timestep, kwargs):
            uncond.append(grabber.features_rows(slice(0, B)).half().cpu())
            cond.append(grabber.features_rows(slice(B, 2 * B)).half().cpu())
            latents.append(kwargs["latents"].detach().cpu())
            return {}

        pre = pipe.transformer.register_forward_pre_hook(grab_noise, with_kwargs=True)
        try:
            images = pipe(prompt_embeds=pe.expand(B, -1, -1), negative_prompt_embeds=npe.expand(B, -1, -1),
                          pooled_prompt_embeds=ppe.expand(B, -1), negative_pooled_prompt_embeds=nppe.expand(B, -1),
                          num_inference_steps=run.STEPS, guidance_scale=run.GUIDANCE, height=SIZE, width=SIZE,
                          generator=[torch.Generator("cpu").manual_seed(s) for s in chunk],
                          callback_on_step_end=on_step, callback_on_step_end_tensor_inputs=["latents"]).images
        finally:
            pre.remove()
        U, C, L = torch.stack(uncond, 1), torch.stack(cond, 1), torch.stack(latents, 1)  # [B, 28, ...]
        for k in range(B):
            out.append({"image": images[k], "feat_uncond": U[k], "feat_cond": C[k],
                        "latents": torch.cat([noise["x"][k:k + 1], L[k]])})  # [29, 16, 64, 64]
    return out


def save(path, obj):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def main():
    assert run.md5(run.CKPT) == run.CKPT_MD5
    for d in ("embeds", "compact", "full", "images"):
        (DATA / d).mkdir(parents=True, exist_ok=True)
    rows = build_prompt_list()
    prompts_path = DATA / "prompts.json"
    if prompts_path.exists():
        assert json.loads(prompts_path.read_text()) == rows, "prompt list changed; refusing to mix datasets"
    else:
        prompts_path.write_text(json.dumps(rows, indent=1))
    n_eval = sum(r["split"] == "eval" for r in rows)
    print(f"{len(rows)} prompts ({n_eval} eval, {len(rows) - n_eval} train), {len(SEEDS)} seeds each at {SIZE}px", flush=True)

    def start_encoder(first_chunk):
        flags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)  # inherited by the venv launcher's child
        return subprocess.Popen([sys.executable, "-u", __file__, "--encode-worker", str(first_chunk)], creationflags=flags)

    encoder = start_encoder(0)
    pipe = StableDiffusion3Pipeline.from_pretrained(
        run.MODEL_ID, text_encoder=None, text_encoder_2=None, text_encoder_3=None,
        tokenizer=None, tokenizer_2=None, tokenizer_3=None, torch_dtype=torch.bfloat16,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    ir_model = RM.load(str(run.IR_PT), device="cuda", med_config=str(run.IR_MED))
    pipe.vae.enable_slicing()  # decode the batch one image at a time
    grabber = BatchRowGrabber(pipe.transformer)
    cache = {}

    try:
        for i, r in enumerate(rows):
            if (DATA / "full" / f"{r['id']}.pt").exists() and (DATA / "compact" / f"{r['id']}.pt").exists():
                continue
            ahead = range(i // CHUNK, i // CHUNK + LOOKAHEAD + 1)
            missing = [c for c in ahead if c * CHUNK < len(rows) and not (DATA / "embeds" / f"chunk_{c:03d}.pt").exists()]
            if missing and encoder.poll() is not None:
                if encoder.returncode != 0:
                    print(f"encoder exited {encoder.returncode}; restarting it", flush=True)
                encoder = start_encoder(i // CHUNK)
            t0 = time.perf_counter()
            cands = run_candidates(pipe, grabber, get_embeds(i, rows, cache), SEEDS, BATCH)
            ir = []
            for s, c in zip(SEEDS, cands):
                img_path = DATA / "images" / f"{r['id']}_seed{s}.png"
                c["image"].save(img_path)
                ir.append(float(ir_model.score(r["prompt"], str(img_path))))
            text_emb = run.blip_text_embedding(ir_model, r["prompt"], "cuda").squeeze(0).cpu()
            meta = {"prompt": r["prompt"], "category": r["category"], "split": r["split"], "seeds": SEEDS,
                    "ir": torch.tensor(ir), "text_emb": text_emb, "size": SIZE, "steps": run.STEPS, "guidance_scale": run.GUIDANCE}
            save(DATA / "full" / f"{r['id']}.pt", {
                **meta,
                "feat_uncond": torch.stack([c["feat_uncond"] for c in cands]),  # [16, 28, 48, 48, 48]
                "feat_cond": torch.stack([c["feat_cond"] for c in cands]),
                "latents": torch.stack([c["latents"] for c in cands]),          # [16, 29, 16, 64, 64]
                "sigmas": pipe.scheduler.sigmas.cpu(), "timesteps": pipe.scheduler.timesteps.cpu(),
            })
            save(DATA / "compact" / f"{r['id']}.pt", {
                **meta, "step": STEP, "t": (STEP + 1) / run.STEPS,
                "feat_uncond": torch.stack([c["feat_uncond"][STEP] for c in cands]),  # [16, 48, 48, 48]
                "feat_cond": torch.stack([c["feat_cond"][STEP] for c in cands]),
            })
            print(f"{r['id']} [{r['category']}] IR {min(ir):+.2f}..{max(ir):+.2f} ({time.perf_counter() - t0:.0f}s)", flush=True)
    finally:
        grabber.handle.remove()
        encoder.kill()


if __name__ == "__main__":
    encode_worker() if "--encode-worker" in sys.argv else main()
