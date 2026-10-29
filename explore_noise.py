"""Visual sweep: how much does throwing noise at a half-formed image change the result?

For each prompt, take the candidate with the best x0_hat ImageReward at step S (t = (S+1)/28 ~ 0.29). From its state
after step S (sigma s, exact velocity v from the saved trajectory): x0_hat = x - s*v, eps_hat = x + (1-s)*v.
  eta sweep (stay at sigma s):   x' = (1-s)*x0_hat + s*(sqrt(1-eta^2)*eps_hat + eta*xi), then denoise from step S+1
  rewind sweep (go back to step j <= S+1, sigma_j >= s):  x' = (1-sigma_j)*x0_hat + sigma_j*xi, then denoise from step j
                 (j = 0 is pure noise: the image is forgotten)
Each setting x N noise draws, finished and scored with ImageReward. Saves data512/figures/noise_explore_<id>.png
and data512/noise_explore.json.
"""
import json

import run
import numpy as np
import torch

from steer import DATA, Sampler, euler, load_embeds

S = 7
ETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
REWIND_STEPS = [8, 6, 4, 2, 0]
DRAWS = 4
PROMPTS = ["eval_0000", "eval_0009", "eval_0016", "train_0034"]  # fruit bowl, a hard non-spatial, a 3d_spatial, a hard train prompt


@torch.no_grad()
def main():
    sm = Sampler()
    rows = json.loads((DATA / "prompts.json").read_text())
    index = {r["id"]: i for i, r in enumerate(rows)}
    summary = {}
    for pid in PROMPTS:
        i, r = index[pid], rows[index[pid]]
        d = torch.load(DATA / "full" / f"{pid}.pt", weights_only=False, mmap=True)
        xh = torch.load(DATA / "xhat0" / f"{pid}.pt", weights_only=False)
        sig, ts, lat = d["sigmas"].float(), d["timesteps"].float(), d["latents"]
        emb = load_embeds(i, sm.device)
        parent = int(np.argmax(xh["score"][xh["steps"].index(S)].numpy()))
        x = lat[parent:parent + 1, S + 1].to(sm.device)
        v = (lat[parent:parent + 1, S + 2].to(sm.device).float() - x.float()) / (sig[S + 2] - sig[S + 1])
        s = sig[S + 1]
        x0_hat, eps_hat = x.float() - s * v, x.float() + (1 - s) * v
        g = torch.Generator("cpu").manual_seed(20_000 + i)
        xi = torch.randn((DRAWS,) + tuple(x.shape[1:]), generator=g).to(sm.device)

        results = {}
        for eta in ETAS:
            xb = ((1 - s) * x0_hat + s * (np.sqrt(1 - eta ** 2) * eps_hat + eta * xi)).to(x.dtype)
            xb, _ = sm.run_steps(xb, S + 1, run.STEPS, sig, ts, emb)
            results[f"eta {eta}"] = sm.score(xb, r["prompt"])
        for j in REWIND_STEPS:
            sj = sig[j]
            xb = ((1 - sj) * x0_hat + sj * xi).to(x.dtype)
            xb, _ = sm.run_steps(xb, j, run.STEPS, sig, ts, emb)
            results[f"rewind to step {j} (sigma {sj:.2f})"] = sm.score(xb, r["prompt"])
        orig = float(d["ir"][parent])
        summary[pid] = {"prompt": r["prompt"], "parent_seed": parent, "parent_final_ir": orig,
                        "prompt_mean_ir": float(d["ir"].mean()), "prompt_best_ir": float(d["ir"].max()),
                        "settings": {k: [round(float(s_), 4) for s_ in v[0]] for k, v in results.items()}}
        print(f"\n{pid}: {r['prompt'][:70]}  (parent seed {parent}: final IR {orig:+.2f}; prompt mean {d['ir'].mean():+.2f}, best {d['ir'].max():+.2f})")
        for k, (sc, _) in results.items():
            print(f"  {k:28s} IR {' '.join(f'{x:+.2f}' for x in sc)}   mean {sc.mean():+.2f}")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        keys = list(results)
        fig, axes = plt.subplots(len(keys), DRAWS + 1, figsize=(2.3 * (DRAWS + 1), 2.45 * len(keys)))
        parent_img = Image.open(DATA / "images" / f"{pid}_seed{parent}.png")
        for row, k in enumerate(keys):
            sc, pils = results[k]
            axes[row, 0].imshow(parent_img); axes[row, 0].axis("off")
            axes[row, 0].set_title(f"original seed {parent}\nIR {orig:+.2f}" if row == 0 else "", fontsize=8)
            axes[row, 0].text(-0.08, 0.5, k, transform=axes[row, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
            for c in range(DRAWS):
                axes[row, c + 1].imshow(pils[c]); axes[row, c + 1].axis("off")
                axes[row, c + 1].set_title(f"IR {sc[c]:+.2f}", fontsize=8, color="tab:green" if sc[c] > orig else "tab:red")
        fig.suptitle(f'{pid}: "{r["prompt"][:90]}"\nnoise applied at t≈{(S + 1) / 28:.2f} to the best candidate (green = beats the original)', fontsize=10)
        fig.tight_layout(); fig.savefig(DATA / "figures" / f"noise_explore_{pid}.png", dpi=90)
        plt.close(fig)
    (DATA / "noise_explore.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
