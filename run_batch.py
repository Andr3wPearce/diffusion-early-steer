"""ProbeSelect SD3-M probe vs ImageReward: 20 prompts x 8 seeds.

Same settings and probe procedure as run.py (which this reuses). Prompt 0 is run.py's prompt, so its
seeds 0-3 reproduce outputs/. The other 19 are sampled (seed 0) from the T2I-CompBench val files.
Outputs go to outputs_20x8/: per-prompt image folders, figures/pNN.png, figures/summary.png, results.json.
"""
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import run  # sets HF cache env, repo path; provides build_probe, blip_text_embedding, decode, md5
import numpy as np
import torch
import ImageReward as RM
from diffusers import StableDiffusion3Pipeline
from scipy.stats import spearmanr

from src.data_gen.utils import ActivationGrabber_SD3

OUT = run.HERE / "outputs_20x8"
SEEDS = list(range(8))
COMPBENCH = run.COMPBENCH_DIR
CATEGORY_COUNTS = {"color": 3, "non_spatial": 3, "complex": 3, "shape": 2, "texture": 2, "spatial": 2, "3d_spatial": 2, "numeracy": 2}


def pick_prompts():
    rng = random.Random(0)
    rows = [{"category": "coco-style (run.py)", "prompt": run.PROMPT}]
    for cat, k in CATEGORY_COUNTS.items():
        lines = [l.strip() for l in (COMPBENCH / f"{cat}_val.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
        rows += [{"category": cat, "prompt": p} for p in rng.sample(lines, k)]
    return rows


def prompt_figure(entry, idx, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    res, n = entry["results"], len(entry["results"])
    fig = plt.figure(figsize=(3.2 * n + 5, 7.6))
    gs = fig.add_gridspec(2, n + 1, width_ratios=[1] * n + [1.6], wspace=0.06, hspace=0.22)
    folder = OUT / f"p{idx:02d}"
    for k, r in enumerate(res):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(Image.open(folder / r["xt_preview"]).resize((384, 384))); ax.axis("off")
        ax.set_title(f"seed {r['seed']}  x_t @ t={r['t_normalized']:.3f}\nprobe {r['probe_score']:.4f} (rank {r['probe_rank']})", fontsize=9)
        ax = fig.add_subplot(gs[1, k])
        ax.imshow(Image.open(folder / r["final_image"]).resize((384, 384))); ax.axis("off")
        ax.set_title(f"final  IR {r['imagereward']:+.3f} (rank {r['imagereward_rank']})", fontsize=9)
    ax = fig.add_subplot(gs[:, n])
    ax.scatter([r["probe_score"] for r in res], [r["imagereward"] for r in res], s=50)
    for r in res:
        ax.annotate(str(r["seed"]), (r["probe_score"], r["imagereward"]), textcoords="offset points", xytext=(5, 3), fontsize=9)
    ax.set_xlabel("probe score at ~20% (relative)")
    ax.set_ylabel("ImageReward (final image)")
    ax.yaxis.set_label_position("right"); ax.yaxis.tick_right()
    ax.set_title(f"Spearman rho = {entry['spearman_rho']:+.3f}\nprobe pick IR rank {entry['probe_pick_ir_rank']} of {n}", fontsize=11)
    ax.grid(alpha=0.3)
    prompt = entry["prompt"] if len(entry["prompt"]) < 110 else entry["prompt"][:107] + "..."
    fig.suptitle(f'p{idx:02d} [{entry["category"]}]  "{prompt}"', fontsize=12)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def summary_figure(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = summary["prompts"]
    n = len(SEEDS)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    ax = axes[0]
    rhos = [e["spearman_rho"] for e in entries]
    ax.bar(range(len(rhos)), rhos, color=["tab:green" if r > 0 else "tab:red" for r in rhos])
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(summary["mean_spearman_rho"], color="tab:blue", ls="--", label=f"mean {summary['mean_spearman_rho']:+.3f}")
    ax.set_xticks(range(len(rhos))); ax.set_xticklabels([f"p{i:02d}" for i in range(len(rhos))], rotation=90, fontsize=8)
    ax.set_ylabel("Spearman rho (probe vs ImageReward, 8 seeds)"); ax.set_title("Per-prompt rank correlation"); ax.legend()
    ax = axes[1]
    counts = np.bincount([e["probe_pick_ir_rank"] for e in entries], minlength=n + 1)[1:]
    ax.bar(range(1, n + 1), counts, color="tab:purple")
    ax.axhline(len(entries) / n, color="k", ls="--", label=f"chance ({len(entries) / n:.1f} each)")
    ax.set_xlabel("ImageReward rank of the probe's top pick (1 = best of 8)"); ax.set_ylabel("prompts")
    ax.set_title(f"Probe top-1 picks the best image in {summary['top1_hits']}/{len(entries)} prompts"); ax.legend()
    ax = axes[2]
    xs = np.arange(len(entries))
    ax.plot(xs, [e["oracle_ir"] for e in entries], "^", color="tab:green", label="best of 8 (oracle)")
    ax.plot(xs, [e["probe_pick_ir"] for e in entries], "o", color="tab:blue", label="probe pick")
    ax.plot(xs, [e["mean_ir"] for e in entries], "_", color="k", ms=14, mew=2, label="random pick (mean)")
    ax.plot(xs, [e["worst_ir"] for e in entries], "v", color="tab:red", label="worst of 8")
    ax.set_xticks(xs); ax.set_xticklabels([f"p{i:02d}" for i in xs], rotation=90, fontsize=8)
    ax.set_ylabel("ImageReward"); ax.legend(fontsize=8)
    ax.set_title(f"Probe pick captures {summary['gain_fraction']:.0%} of the best-of-8 gain over random")
    fig.suptitle(f"ProbeSelect SD3-M probe at t~0.21 vs final ImageReward: {len(entries)} prompts x {n} seeds", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def summarize(entries):
    s = {
        "num_prompts": len(entries), "seeds": SEEDS,
        "mean_spearman_rho": float(np.mean([e["spearman_rho"] for e in entries])),
        "median_spearman_rho": float(np.median([e["spearman_rho"] for e in entries])),
        "prompts_with_positive_rho": int(sum(e["spearman_rho"] > 0 for e in entries)),
        "top1_hits": int(sum(e["probe_pick_ir_rank"] == 1 for e in entries)),
        "top1_chance": 1 / len(SEEDS),
        "mean_probe_pick_ir_rank": float(np.mean([e["probe_pick_ir_rank"] for e in entries])),
        "mean_ir_probe_pick": float(np.mean([e["probe_pick_ir"] for e in entries])),
        "mean_ir_random": float(np.mean([e["mean_ir"] for e in entries])),
        "mean_ir_oracle": float(np.mean([e["oracle_ir"] for e in entries])),
    }
    s["gain_fraction"] = (s["mean_ir_probe_pick"] - s["mean_ir_random"]) / (s["mean_ir_oracle"] - s["mean_ir_random"])
    probe_all = np.array([r["probe_score"] for e in entries for r in e["results"]])
    s["probe_score_range"] = [float(probe_all.min()), float(probe_all.max())]
    # Sign-flip test on per-prompt rho: how often would random signs give a mean this large?
    rng = np.random.default_rng(0)
    rhos = np.array([e["spearman_rho"] for e in entries])
    flips = rng.choice([-1, 1], size=(100000, len(rhos))) * np.abs(rhos)
    s["mean_rho_p_value_signflip"] = float((flips.mean(1) >= rhos.mean()).mean())
    s["prompts"] = entries
    return s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    device = "cuda"
    assert run.md5(run.CKPT) == run.CKPT_MD5, "probe checkpoint md5 mismatch"
    prompts = pick_prompts()
    (OUT / "prompts.json").write_text(json.dumps(prompts, indent=1))

    # CPU text encoding has segfaulted intermittently here, so it runs in a child process (retried) that
    # saves the embeddings; a crash there can't take down the generation run.
    emb_path = OUT / "prompt_embeds.pt"
    for attempt in range(1, 4):
        if emb_path.exists():
            break
        rc = subprocess.run([sys.executable, "-u", __file__, "--encode-only"]).returncode
        print(f"encode subprocess attempt {attempt} exited {rc}", flush=True)
    saved = torch.load(emb_path, weights_only=True)
    assert saved["prompts"] == [p["prompt"] for p in prompts], "cached embeddings are for different prompts"
    embeds = saved["embeds"]

    pipe = StableDiffusion3Pipeline.from_pretrained(
        run.MODEL_ID, text_encoder=None, text_encoder_2=None, text_encoder_3=None,
        tokenizer=None, tokenizer_2=None, tokenizer_3=None, torch_dtype=torch.bfloat16,
    ).to(device)
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    probe = run.build_probe(device)
    ir_model = RM.load(str(run.IR_PT), device=device, med_config=str(run.IR_MED))
    grabber = ActivationGrabber_SD3(pipe.transformer)

    results_path = OUT / "results.json"
    done = {e["index"]: e for e in json.loads(results_path.read_text())["prompts"]} if results_path.exists() else {}
    entries = []
    for idx, (p, (pe, npe, ppe, nppe)) in enumerate(zip(prompts, embeds)):
        if idx in done and done[idx]["prompt"] == p["prompt"]:
            entries.append(done[idx]); continue
        folder = OUT / f"p{idx:02d}"
        folder.mkdir(exist_ok=True)
        emb = dict(prompt_embeds=pe.to(device), negative_prompt_embeds=npe.to(device),
                   pooled_prompt_embeds=ppe.to(device), negative_pooled_prompt_embeds=nppe.to(device))
        text_emb = run.blip_text_embedding(ir_model, p["prompt"], device)
        res = []
        t1 = time.perf_counter()
        for seed in SEEDS:
            rec = {}

            def on_step(_pipe, i, timestep, kwargs):
                if "probe_score" not in rec and (i + 1) / run.STEPS >= 0.2:
                    feats = grabber.get_pca_feature().to(device).float()
                    t_norm = torch.full((feats.shape[0],), (i + 1) / run.STEPS, device=device)
                    with torch.no_grad():
                        score, _ = probe(feats, t_norm, text_emb=text_emb)
                    rec.update(probe_score=float(score), step_index=i, t_normalized=(i + 1) / run.STEPS)
                    run.decode(_pipe, kwargs["latents"]).save(folder / f"xt_seed{seed}.png")
                grabber.latest_feats.clear()
                return {}

            image = pipe(
                **emb, num_inference_steps=run.STEPS, guidance_scale=run.GUIDANCE, height=run.SIZE, width=run.SIZE,
                generator=torch.Generator("cpu").manual_seed(seed),
                callback_on_step_end=on_step, callback_on_step_end_tensor_inputs=["latents"],
            ).images[0]
            torch.cuda.empty_cache()
            image.save(folder / f"final_seed{seed}.png")
            rec["imagereward"] = float(ir_model.score(p["prompt"], str(folder / f"final_seed{seed}.png")))
            res.append({"seed": seed, **rec, "xt_preview": f"xt_seed{seed}.png", "final_image": f"final_seed{seed}.png"})

        for key, rank_key in (("probe_score", "probe_rank"), ("imagereward", "imagereward_rank")):
            for rank, r in enumerate(sorted(res, key=lambda r: -r[key]), 1):
                r[rank_key] = rank
        irs = [r["imagereward"] for r in res]
        pick = min(res, key=lambda r: r["probe_rank"])
        entry = {
            "index": idx, **p, "results": res,
            "spearman_rho": float(spearmanr([r["probe_score"] for r in res], irs).statistic),
            "probe_pick_seed": pick["seed"], "probe_pick_ir": pick["imagereward"], "probe_pick_ir_rank": pick["imagereward_rank"],
            "oracle_ir": max(irs), "mean_ir": float(np.mean(irs)), "worst_ir": min(irs),
        }
        entries.append(entry)
        prompt_figure(entry, idx, OUT / "figures" / f"p{idx:02d}.png")
        results_path.write_text(json.dumps(summarize(entries), indent=1))  # checkpoint after every prompt
        print(f"p{idx:02d} [{p['category']}] rho {entry['spearman_rho']:+.3f}, probe pick IR rank {entry['probe_pick_ir_rank']}/8 "
              f"({time.perf_counter() - t1:.0f}s)", flush=True)

    grabber.close()
    summary = summarize(entries)
    summary["environment"] = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                              "notes": "prompts encoded on CPU; VAE tiling enabled; probe run in float32"}
    results_path.write_text(json.dumps(summary, indent=1))
    summary_figure(summary, OUT / "figures" / "summary.png")
    print(f"\nmean rho {summary['mean_spearman_rho']:+.3f} (median {summary['median_spearman_rho']:+.3f}, "
          f"{summary['prompts_with_positive_rho']}/{len(entries)} positive, sign-flip p={summary['mean_rho_p_value_signflip']:.3f}); "
          f"top-1 {summary['top1_hits']}/{len(entries)} (chance {len(entries) / 8:.1f}); "
          f"IR pick {summary['mean_ir_probe_pick']:+.3f} vs random {summary['mean_ir_random']:+.3f} vs oracle {summary['mean_ir_oracle']:+.3f} "
          f"({summary['gain_fraction']:.0%} of possible gain)")


def encode_only():
    prompts = pick_prompts()
    t0 = time.perf_counter()
    te_pipe = StableDiffusion3Pipeline.from_pretrained(run.MODEL_ID, transformer=None, vae=None, torch_dtype=torch.bfloat16)
    # encode_prompt is not wrapped in no_grad (only the pipeline's __call__ is). Without this, every kept
    # embedding holds T5's autograd graph (~4.4 GB per prompt) until RAM runs out and the process crashes.
    with torch.no_grad():
        embeds = [te_pipe.encode_prompt(p["prompt"], None, None, device="cpu", do_classifier_free_guidance=True) for p in prompts]
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "prompt_embeds.pt.tmp"
    torch.save({"prompts": [p["prompt"] for p in prompts], "embeds": [tuple(t.contiguous() for t in e) for e in embeds]}, tmp)
    tmp.replace(OUT / "prompt_embeds.pt")
    print(f"{len(prompts)} prompts encoded on CPU in {time.perf_counter() - t0:.0f}s", flush=True)
    os._exit(0)  # skip interpreter teardown of the text encoders


if __name__ == "__main__":
    encode_only() if "--encode-only" in sys.argv else main()
