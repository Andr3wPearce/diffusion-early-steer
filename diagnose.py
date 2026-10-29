"""Why does the released SD3-M probe barely track ImageReward? Re-run the first 10 prompts x 8 seeds
(trajectories are deterministic, so the features are exactly those of the finished images in outputs_20x8/)
and score each candidate under several variants:
  - step i in STEPS_TO_TEST (t from 0.21 to 0.82)
  - CFG row: unconditional (what [0::2] picks at batch 1) vs conditional
  - time label: (i+1)/28 (ProbeSelect_SD3.__call__) vs i/28 (their data generator's record['t'])
Denoising stops after the last tested step. ImageReward labels come from outputs_20x8/results.json.
"""
import json
from math import sqrt
from pathlib import Path

import run  # env, repo path, probe/text-embedding helpers
import numpy as np
import torch
import torch.nn.functional as F
import ImageReward as RM
from diffusers import StableDiffusion3Pipeline
from scipy.stats import spearmanr

from src.data_gen.utils import pca_channel_per_sample

OUT = run.HERE / "outputs_20x8"
NUM_PROMPTS = 10
STEPS_TO_TEST = [5, 8, 11, 16, 22]
ROWS = {"uncond": 0, "cond": 1}


class Stop(Exception):
    pass


class RowGrabber:
    """ActivationGrabber_SD3's processing (block 20, unpatchify, resize to 48x48), keeping both CFG rows."""

    def __init__(self, transformer):
        self.patch = transformer.config.patch_size
        self.latest = None
        self.handle = transformer.transformer_blocks[20].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.latest = output[1].detach()

    def features(self, row):
        x = self.latest[row:row + 1]
        bs, n, c = x.shape
        p, new_c = self.patch, c // self.patch // self.patch
        h = w = int(sqrt(n))
        x = x.view(bs, h, w, p, p, new_c).permute(0, 5, 1, 3, 2, 4).reshape(bs, new_c, h * p, w * p)
        x = F.interpolate(x, size=(48, 48), mode="bilinear", align_corners=False)
        return pca_channel_per_sample(x, low_c=48)


def main():
    device = "cuda"
    batch = json.loads((OUT / "results.json").read_text())
    entries = batch["prompts"][:NUM_PROMPTS]
    saved = torch.load(OUT / "prompt_embeds.pt", weights_only=True)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        run.MODEL_ID, text_encoder=None, text_encoder_2=None, text_encoder_3=None,
        tokenizer=None, tokenizer_2=None, tokenizer_3=None, torch_dtype=torch.bfloat16,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    probe = run.build_probe(device)
    ir_model = RM.load(str(run.IR_PT), device=device, med_config=str(run.IR_MED))
    grabber = RowGrabber(pipe.transformer)

    # scores[variant] -> list over prompts of list over seeds
    variants = [(i, row, tl) for i in STEPS_TO_TEST for row in ROWS for tl in ("i+1", "i")]
    scores = {v: [] for v in variants}
    feat_stats, check = [], []
    for e in entries:
        idx = e["index"]
        assert saved["prompts"][idx] == e["prompt"]
        pe, npe, ppe, nppe = (t.to(device) for t in saved["embeds"][idx])
        text_emb = run.blip_text_embedding(ir_model, e["prompt"], device)
        per_seed = {v: [] for v in variants}
        for r in e["results"]:
            def on_step(_pipe, i, timestep, kwargs):
                if i in STEPS_TO_TEST:
                    for row_name, row in ROWS.items():
                        feats = grabber.features(row).to(device).float()
                        if i == 5 and row_name == "uncond":
                            feat_stats.append(feats.flatten())
                        for tl in ("i+1", "i"):
                            t = torch.full((1,), (i + 1 if tl == "i+1" else i) / run.STEPS, device=device)
                            with torch.no_grad():
                                s, _ = probe(feats, t, text_emb=text_emb)
                            per_seed[(i, row_name, tl)].append(float(s))
                if i >= max(STEPS_TO_TEST):
                    raise Stop
                return {}

            try:
                pipe(prompt_embeds=pe, negative_prompt_embeds=npe, pooled_prompt_embeds=ppe, negative_pooled_prompt_embeds=nppe,
                     num_inference_steps=run.STEPS, guidance_scale=run.GUIDANCE, height=run.SIZE, width=run.SIZE,
                     generator=torch.Generator("cpu").manual_seed(r["seed"]),
                     callback_on_step_end=on_step, callback_on_step_end_tensor_inputs=["latents"])
            except Stop:
                pass
            check.append(abs(per_seed[(5, "uncond", "i+1")][-1] - r["probe_score"]))
        for v in variants:
            scores[v].append(per_seed[v])
        print(f"p{idx:02d} done; max |score - batch run| so far {max(check):.2e}", flush=True)

    irs = [[r["imagereward"] for r in e["results"]] for e in entries]

    def evaluate(per_prompt):
        rhos = [spearmanr(s, y).statistic for s, y in zip(per_prompt, irs)]
        top1 = [int(np.argmax(s) == np.argmax(y)) for s, y in zip(per_prompt, irs)]
        pick_ir = [y[int(np.argmax(s))] for s, y in zip(per_prompt, irs)]
        spread = [float(np.std(s)) for s in per_prompt]
        return {"mean_rho": float(np.mean(rhos)), "rhos": [float(x) for x in rhos], "top1_hits": int(sum(top1)),
                "mean_pick_ir": float(np.mean(pick_ir)), "mean_within_prompt_std": float(np.mean(spread)),
                "mean_score": float(np.mean(per_prompt))}

    table = {f"step{i}_t{(i + 1) / run.STEPS:.3f}_{row}_tlabel({tl})": evaluate(scores[(i, row, tl)]) for (i, row, tl) in variants}

    # Sensitivity: random features with the same mean/std as real t~0.21 uncond features.
    allf = torch.cat(feat_stats)
    mu, sd = allf.mean().item(), allf.std().item()
    rand_scores = []
    with torch.no_grad():
        text_emb = run.blip_text_embedding(ir_model, entries[0]["prompt"], device)
        for k in range(32):
            g = torch.Generator(device).manual_seed(k)
            f = torch.randn(1, 48, 48, 48, device=device, generator=g) * sd + mu
            rand_scores.append(float(probe(f, torch.full((1,), 6 / 28, device=device), text_emb=text_emb)[0]))
    real = [s for p in scores[(5, "uncond", "i+1")] for s in p]

    out = {
        "num_prompts": len(entries), "seeds_per_prompt": len(entries[0]["results"]),
        "reproduces_batch_scores_max_abs_diff": max(check),
        "random_baseline": {"mean_ir_random_pick": float(np.mean([np.mean(y) for y in irs])),
                            "mean_ir_oracle": float(np.mean([max(y) for y in irs])),
                            "top1_hits_expected_by_chance": len(entries) / len(entries[0]["results"])},
        "sensitivity_t0.214_uncond": {
            "real_features": {"mean": float(np.mean(real)), "std": float(np.std(real)), "min": float(np.min(real)), "max": float(np.max(real))},
            "random_features_matched_stats": {"mean": float(np.mean(rand_scores)), "std": float(np.std(rand_scores)),
                                              "min": float(np.min(rand_scores)), "max": float(np.max(rand_scores))},
            "feature_mean": mu, "feature_std": sd},
        "variants": table,
    }
    (OUT / "diagnostics.json").write_text(json.dumps(out, indent=1))
    print("\nvariant                                       mean_rho  top1  pick_IR  score_mean  within_std")
    for k, v in table.items():
        print(f"{k:45s} {v['mean_rho']:+.3f}   {v['top1_hits']:2d}   {v['mean_pick_ir']:+.3f}   {v['mean_score']:.4f}     {v['mean_within_prompt_std']:.5f}")
    rb = out["random_baseline"]
    print(f"\nrandom pick IR {rb['mean_ir_random_pick']:+.3f}, oracle {rb['mean_ir_oracle']:+.3f}, top1 by chance {rb['top1_hits_expected_by_chance']:.2f}")
    s = out["sensitivity_t0.214_uncond"]
    print(f"real feats score {s['real_features']['mean']:.4f} +/- {s['real_features']['std']:.5f}; "
          f"random feats {s['random_features_matched_stats']['mean']:.4f} +/- {s['random_features_matched_stats']['std']:.5f}")


if __name__ == "__main__":
    main()
