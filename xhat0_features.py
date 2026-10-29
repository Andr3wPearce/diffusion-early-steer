"""Per-candidate x0_hat features for combining with probes (incremental; rerun as data512 grows).

For each prompt in data512/full and each step in STEPS: x0_hat = x_i - sigma_i * v_i (v_i exact from consecutive
saved latents, see xhat0_baseline.py), decoded with the VAE and passed through ImageReward. Saves the ImageReward
score and ImageReward's 768-d joint image-text feature (the input to its reward MLP) per candidate:
  data512/xhat0/<id>.pt: {"steps": [...], "score": [n_steps, 16], "emb": [n_steps, 16, 768] fp16, "ir": [16], ...}
"""
import time

import run
import torch
import ImageReward as RM
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

DATA = run.HERE / "data512"
STEPS = [3, 4, 5, 6, 7, 8, 10, 13]  # t = (i+1)/28 = 0.14, 0.18, 0.21, 0.25, 0.29, 0.32, 0.39, 0.50


@torch.no_grad()
def reward_and_embedding(model, prompt, images):
    """ImageReward.score's computation, also returning the joint feature fed to the reward MLP."""
    text = model.blip.tokenizer(prompt, padding="max_length", truncation=True, max_length=35, return_tensors="pt").to(model.device)
    image = torch.stack([model.preprocess(im) for im in images]).to(model.device)
    image_embeds = model.blip.visual_encoder(image)
    atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=model.device)
    n = len(images)
    emb = model.blip.text_encoder(text.input_ids.expand(n, -1), attention_mask=text.attention_mask.expand(n, -1),
                                  encoder_hidden_states=image_embeds, encoder_attention_mask=atts,
                                  return_dict=True).last_hidden_state[:, 0, :].float()
    score = ((model.mlp(emb) - model.mean) / model.std).squeeze(-1)
    return score.cpu(), emb.half().cpu()


@torch.no_grad()
def main():
    device = "cuda"
    out_dir = DATA / "xhat0"
    out_dir.mkdir(exist_ok=True)
    vae = AutoencoderKL.from_pretrained(run.MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16).to(device)
    vae.enable_slicing()
    proc = VaeImageProcessor(vae_scale_factor=8)
    ir_model = RM.load(str(run.IR_PT), device=device, med_config=str(run.IR_MED))

    files = [f for f in sorted((DATA / "full").glob("*.pt")) if not (out_dir / f.name).exists()]
    print(f"{len(files)} prompts to process", flush=True)
    t0 = time.perf_counter()
    for n, f in enumerate(files):
        d = torch.load(f, weights_only=False, mmap=True)
        lat, sig = d["latents"], d["sigmas"].float()
        scores, embs = [], []
        for s in STEPS:
            x, x_next = lat[:, s].to(device).float(), lat[:, s + 1].to(device).float()
            x0 = x - sig[s] * (x_next - x) / (sig[s + 1] - sig[s])
            img = vae.decode((x0 / vae.config.scaling_factor + vae.config.shift_factor).to(vae.dtype), return_dict=False)[0]
            sc, em = reward_and_embedding(ir_model, d["prompt"], proc.postprocess(img, output_type="pil"))
            scores.append(sc); embs.append(em)
        tmp = (out_dir / f.name).with_suffix(".tmp")
        torch.save({"steps": STEPS, "score": torch.stack(scores), "emb": torch.stack(embs), "ir": d["ir"],
                    "prompt": d["prompt"], "category": d["category"], "split": d["split"]}, tmp)
        tmp.replace(out_dir / f.name)
        if (n + 1) % 5 == 0:
            el = time.perf_counter() - t0
            print(f"{n + 1}/{len(files)} prompts ({el / (n + 1):.0f}s each, ~{el / (n + 1) * (len(files) - n - 1) / 60:.0f} min left)", flush=True)
    print("XHAT0 FEATURES DONE", flush=True)


if __name__ == "__main__":
    main()
