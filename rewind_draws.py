"""Many noise draws on the SAME image: how far can rewinding move one candidate's final ImageReward?

For each prompt in PROMPTS, take the candidate ranked worst by x0_hat ImageReward at step J (what the rewind policy
would target). For each depth j, draw DRAWS independent noises: x' = (1 - sigma_j) * x0_hat_J + sigma_j * xi, denoise
from step j to the end, score with ImageReward. Saves data512/figures/rewind_draws_<id>.png (score spread per depth +
8 example images per depth) and data512/rewind_draws.json.
"""
import json

import run
import numpy as np
import torch

from rewind import DEPTHS, J
from steer import DATA, Sampler, load_embeds

PROMPTS = ["eval_0000", "eval_0016", "eval_0009"]  # fruit bowl, dog, metal chair / laptop (hard)
DRAWS = 16
SHOW = 8


@torch.no_grad()
def main():
    sm = Sampler()
    rows = json.loads((DATA / "prompts.json").read_text())
    index = {r["id"]: i for i, r in enumerate(rows)}
    out = {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    for pid in PROMPTS:
        i, r = index[pid], rows[index[pid]]
        d = torch.load(DATA / "full" / f"{pid}.pt", weights_only=False, mmap=True)
        xh = torch.load(DATA / "xhat0" / f"{pid}.pt", weights_only=False)
        sig, ts, lat = d["sigmas"].float(), d["timesteps"].float(), d["latents"]
        emb = load_embeds(i, sm.device)
        k = int(np.argmin(xh["score"][xh["steps"].index(J)].numpy()))
        x = lat[k:k + 1, J].to(sm.device)
        v = (lat[k:k + 1, J + 1].to(sm.device).float() - x.float()) / (sig[J + 1] - sig[J])
        x0_hat = x.float() - sig[J] * v
        orig = float(d["ir"][k])
        res = {}
        for j in DEPTHS:
            g = torch.Generator("cpu").manual_seed(40_000 + 100 * i + j)
            xi = torch.randn((DRAWS,) + tuple(x0_hat.shape[1:]), generator=g).to(sm.device)
            xb = ((1 - sig[j]) * x0_hat + sig[j] * xi).to(torch.bfloat16)
            xb, _ = sm.run_steps(xb, j, run.STEPS, sig, ts, emb)
            res[j] = sm.score(xb, r["prompt"])
        scores = {j: res[j][0] for j in DEPTHS}
        out[pid] = {"prompt": r["prompt"], "candidate_seed": k, "original_final_ir": orig,
                    "prompt_mean_ir": float(d["ir"].mean()), "prompt_best_ir": float(d["ir"].max()),
                    "draws": {str(j): [round(float(s), 4) for s in scores[j]] for j in DEPTHS}}
        print(f"\n{pid} seed {k} (original final IR {orig:+.2f}; prompt mean {d['ir'].mean():+.2f}, best of 16 {d['ir'].max():+.2f})")
        for j in DEPTHS:
            s = scores[j]
            print(f"  step {j} (sigma {sig[j]:.2f}): mean {s.mean():+.2f}  min {s.min():+.2f}  max {s.max():+.2f}  "
                  f"beat original {int((s > orig).sum())}/{DRAWS}  beat prompt best {int((s > d['ir'].max()).sum())}/{DRAWS}")

        fig = plt.figure(figsize=(3.2 + 1.6 * SHOW, 1.75 * len(DEPTHS) + 1.4))
        gs = fig.add_gridspec(len(DEPTHS), SHOW + 2, width_ratios=[2.2, 1] + [1] * SHOW, wspace=0.05, hspace=0.25)
        ax = fig.add_subplot(gs[:, 0])
        for row, j in enumerate(DEPTHS):
            ax.scatter(scores[j], np.full(DRAWS, row) + np.random.default_rng(j).uniform(-0.15, 0.15, DRAWS), s=14, alpha=0.75)
        ax.axvline(orig, color="tab:red", ls="--", label=f"original {orig:+.2f}")
        ax.axvline(float(d["ir"].mean()), color="gray", ls=":", label=f"prompt mean {d['ir'].mean():+.2f}")
        ax.axvline(float(d["ir"].max()), color="tab:green", ls=":", label=f"best of 16 {d['ir'].max():+.2f}")
        ax.set_yticks(range(len(DEPTHS))); ax.set_yticklabels([f"step {j}\nσ={sig[j]:.2f}" for j in DEPTHS], fontsize=8)
        ax.invert_yaxis(); ax.set_xlabel("final ImageReward"); ax.legend(fontsize=7, loc="lower left"); ax.grid(alpha=0.3)
        ax.set_title(f"{DRAWS} draws per depth", fontsize=9)
        orig_img = Image.open(DATA / "images" / f"{pid}_seed{k}.png")
        for row, j in enumerate(DEPTHS):
            a = fig.add_subplot(gs[row, 1]); a.imshow(orig_img); a.axis("off")
            if row == 0:
                a.set_title(f"original\n{orig:+.2f}", fontsize=7)
            sc, pils = res[j]
            for c in range(SHOW):
                a = fig.add_subplot(gs[row, c + 2]); a.imshow(pils[c]); a.axis("off")
                a.set_title(f"{sc[c]:+.2f}", fontsize=7, color="tab:green" if sc[c] > orig else "tab:red")
        fig.suptitle(f'{pid}: "{r["prompt"][:90]}" — worst candidate at t=0.5 (seed {k}) rewound to each depth', fontsize=10)
        fig.savefig(DATA / "figures" / f"rewind_draws_{pid}.png", dpi=85, bbox_inches="tight")
        plt.close(fig)
    (DATA / "rewind_draws.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
