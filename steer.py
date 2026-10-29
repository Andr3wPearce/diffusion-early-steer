"""Steering with noise at t ~ 0.3, vs early selection, on SD 3.5 Medium @512 (data512 trajectories).

Per prompt (16 saved candidates, deterministic ODE trajectories):
  pick top K by x0_hat ImageReward at step S1 (t = 8/28 = 0.29), branch each into B by re-noising the state after
  step S1 (x' = (1-s)*x0_hat + s*(sqrt(1-eta^2)*eps_hat + eta*xi), eta = 0 reproduces the original path),
  denoise every branch through step S2 (t = 0.5), pick the best branch by x0_hat ImageReward at S2, finish it,
  score the final image with ImageReward.
Baselines come from saved data (trajectories are deterministic): random single sample, select-at-S1, select-at-S2
(same compute as steering), oracle best-of-16 by final ImageReward.
Own sampler = SD3 pipeline loop (CFG, FlowMatch Euler on the saved sigma schedule); --validate checks eta = 0 reproduces
saved final latents. Outputs data512/steer/<id>.json, data512/steer_summary.json, data512/figures/steering.png.
"""
import argparse
import json
import time

import run
import numpy as np
import torch
import ImageReward as RM
from diffusers import AutoencoderKL, SD3Transformer2DModel
from diffusers.image_processor import VaeImageProcessor

from xhat0_features import reward_and_embedding

DATA = run.HERE / "data512"
S1, S2 = 7, 13          # decide after the step-7 forward (t = 8/28), select branches at step 13 (t = 14/28)
K, B = 4, 4             # top-K candidates x B branches = 16 branches
ETAS = [0.5, 1.0]
CHUNK = 50


class Sampler:
    def __init__(self, device="cuda"):
        self.device = device
        self.tr = SD3Transformer2DModel.from_pretrained(run.MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16).to(device)
        self.vae = AutoencoderKL.from_pretrained(run.MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16).to(device)
        self.vae.enable_slicing()
        self.proc = VaeImageProcessor(vae_scale_factor=8)
        self.ir = RM.load(str(run.IR_PT), device=device, med_config=str(run.IR_MED))

    @torch.no_grad()
    def velocity(self, x, t, emb):
        """CFG velocity exactly as StableDiffusion3Pipeline: batch [uncond, cond], uncond + g * (cond - uncond)."""
        pe, npe, ppe, nppe = emb
        n = x.shape[0]
        out = self.tr(hidden_states=torch.cat([x, x]), timestep=t.expand(2 * n),
                      encoder_hidden_states=torch.cat([npe.expand(n, -1, -1), pe.expand(n, -1, -1)]),
                      pooled_projections=torch.cat([nppe.expand(n, -1), ppe.expand(n, -1)]), return_dict=False)[0]
        u, c = out.chunk(2)
        return u + run.GUIDANCE * (c - u)

    @torch.no_grad()
    def run_steps(self, x, start, stop, sig, ts, emb):
        """Denoise steps start..stop-1 with FlowMatchEulerDiscreteScheduler.step's arithmetic. Bit-exact detail:
        (sigma_next - sigma) is a 0-dim fp32 tensor, so its product with the bf16 velocity stays bf16 before the
        fp32 add. Returns the final x and the velocity of the last step."""
        v = None
        for j in range(start, stop):
            v = self.velocity(x, ts[j].to(self.device), emb)
            x = euler(x, v, sig, j)
        return x, v

    @torch.no_grad()
    def score(self, latents, prompt):
        img = self.vae.decode((latents / self.vae.config.scaling_factor + self.vae.config.shift_factor).to(self.vae.dtype), return_dict=False)[0]
        pils = self.proc.postprocess(img, output_type="pil")
        s, _ = reward_and_embedding(self.ir, prompt, pils)
        return s.numpy().astype(np.float64), pils


def euler(x, v, sig, j):
    """One FlowMatch Euler step exactly as the scheduler computes it (see Sampler.run_steps)."""
    dt = (sig[j + 1] - sig[j]).to(device=v.device, dtype=torch.float32)
    return (x.float() + dt * v).to(x.dtype)


def load_embeds(index, device):
    chunk = torch.load(DATA / "embeds" / f"chunk_{index // CHUNK:03d}.pt", weights_only=True)
    return tuple(t.to(device) for t in chunk["embeds"][index % CHUNK])


def validate(sm, rows):
    r = rows[0]
    d = torch.load(DATA / "full" / f"{r['id']}.pt", weights_only=False, mmap=True)
    sig, ts = d["sigmas"].float(), d["timesteps"].float()
    emb = load_embeds(0, sm.device)
    x = d["latents"][:4, S1 + 1].to(sm.device)
    x, _ = sm.run_steps(x, S1 + 1, run.STEPS, sig, ts, emb)
    ref = d["latents"][:4, -1].to(sm.device)
    s, _ = sm.score(x, d["prompt"])
    print(f"validate {r['id']}: final latents bit-identical: {torch.equal(x, ref)}; max |diff| = {(x.float() - ref.float()).abs().max().item():.4f} "
          f"(latent std {ref.float().std().item():.2f}); IR ours {np.round(s, 3)} vs saved {np.round(d['ir'][:4].numpy(), 3)}")


def steer_prompt(sm, idx, r):
    d = torch.load(DATA / "full" / f"{r['id']}.pt", weights_only=False, mmap=True)
    xh = torch.load(DATA / "xhat0" / f"{r['id']}.pt", weights_only=False)
    sig, ts, lat, final_ir = d["sigmas"].float(), d["timesteps"].float(), d["latents"], d["ir"].numpy().astype(np.float64)
    emb = load_embeds(idx, sm.device)
    s1_scores = xh["score"][xh["steps"].index(S1)].numpy()
    s2_scores = xh["score"][xh["steps"].index(S2)].numpy()
    top = np.argsort(-s1_scores)[:K]
    out = {"id": r["id"], "prompt": r["prompt"], "category": r["category"], "split": r["split"],
           "random": float(final_ir.mean()), "oracle16": float(final_ir.max()),
           "select_s1": float(final_ir[int(np.argmax(s1_scores))]), "select_s2": float(final_ir[int(np.argmax(s2_scores))]),
           "top_k_original_best": float(final_ir[top].max()), "top_k": top.tolist(), "steer": {}}
    # State after step S1 and its velocity (exact from the saved trajectory), for the top-K candidates.
    x = lat[top, S1 + 1].to(sm.device)
    v = (lat[top, S1 + 2].to(sm.device).float() - x.float()) / (sig[S1 + 2] - sig[S1 + 1])
    s = sig[S1 + 1]
    x0_hat, eps_hat = x.float() - s * v, x.float() + (1 - s) * v
    for eta in ETAS:
        g = torch.Generator("cpu").manual_seed(10_000 + idx)
        xi = torch.randn((K * B,) + tuple(x.shape[1:]), generator=g).to(sm.device)
        rep = lambda t: t.repeat_interleave(B, 0)
        xb = ((1 - s) * rep(x0_hat) + s * (np.sqrt(1 - eta ** 2) * rep(eps_hat) + eta * xi)).to(x.dtype)
        # Denoise steps S1+1 .. S2-1, then the S2 forward gives x0_hat at t = (S2+1)/28 for selection.
        xb, _ = sm.run_steps(xb, S1 + 1, S2, sig, ts, emb)
        v2 = sm.velocity(xb, ts[S2].to(sm.device), emb)
        sel_scores, _ = sm.score((xb.float() - sig[S2] * v2.float()).to(xb.dtype), r["prompt"])
        best = int(np.argmax(sel_scores))
        xf = euler(xb[best:best + 1], v2[best:best + 1], sig, S2)
        xf, _ = sm.run_steps(xf, S2 + 1, run.STEPS, sig, ts, emb)
        final_score, pils = sm.score(xf, r["prompt"])
        pils[0].save(DATA / "steer" / f"{r['id']}_eta{eta}.png")
        out["steer"][str(eta)] = {"final_ir": float(final_score[0]), "branch": best, "parent_seed": int(top[best // B]),
                                  "branch_scores_at_s2": sel_scores.round(4).tolist()}
    return out


def summarize(results):
    n = len(results)
    methods = {"random single sample": ("random", 1.0),
               f"select @t={(S1 + 1) / 28:.2f}": ("select_s1", 16 * (S1 + 2) / 28 + (28 - S1 - 2) / 28),
               f"select @t={(S2 + 1) / 28:.2f}": ("select_s2", 16 * (S2 + 1) / 28 + (28 - S2 - 1) / 28)}
    steer_cost = 16 * (S1 + 2) / 28 + K * B * (S2 + 1 - S1 - 1) / 28 + (28 - S2 - 1) / 28
    rows = {}
    for name, (key, cost) in methods.items():
        v = np.array([r[key] for r in results]); rows[name] = (v, cost)
    for eta in ETAS:
        v = np.array([r["steer"][str(eta)]["final_ir"] for r in results]); rows[f"steer eta={eta}"] = (v, steer_cost)
    rows["oracle best-of-16 (final IR)"] = (np.array([r["oracle16"] for r in results]), 16.0)
    base = rows[f"select @t={(S2 + 1) / 28:.2f}"][0]
    summary = {}
    print(f"\n{n} prompts. Final ImageReward (mean ± se), compute in full generations:")
    for name, (v, cost) in rows.items():
        diff = v - base
        summary[name] = {"mean_ir": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(n)), "compute": cost,
                         "vs_select_s2_mean": float(diff.mean()), "vs_select_s2_se": float(diff.std(ddof=1) / np.sqrt(n))}
        print(f"  {name:32s} {v.mean():+.3f} ± {v.std(ddof=1) / np.sqrt(n):.3f}   compute {cost:4.1f}   "
              f"vs select@0.5: {diff.mean():+.3f} ± {diff.std(ddof=1) / np.sqrt(n):.3f}")
    return summary, rows


def figure(rows, path, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, (v, cost) in rows.items():
        ax.errorbar(cost, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), fmt="o", ms=8, capsize=4, label=name)
    ax.set_xlabel("compute (full generations)"); ax.set_ylabel(f"final ImageReward (mean ± se over {n} prompts)")
    ax.set_title("Steering with noise at t≈0.29 vs early selection (SD 3.5 Medium @512, 16 candidates)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--num-train", type=int, default=25, help="train prompts to include after the 25 eval prompts")
    args = ap.parse_args()
    rows = json.loads((DATA / "prompts.json").read_text())
    sm = Sampler()
    if args.validate:
        validate(sm, rows)
        return
    (DATA / "steer").mkdir(exist_ok=True)
    have = {p.stem for p in (DATA / "xhat0").glob("*.pt")}
    todo = [(i, r) for i, r in enumerate(rows) if r["id"] in have]
    todo = [x for x in todo if x[1]["split"] == "eval"] + [x for x in todo if x[1]["split"] == "train"][:args.num_train]
    results = []
    t0 = time.perf_counter()
    for n, (i, r) in enumerate(todo):
        path = DATA / "steer" / f"{r['id']}.json"
        if path.exists():
            results.append(json.loads(path.read_text())); continue
        res = steer_prompt(sm, i, r)
        path.write_text(json.dumps(res, indent=1)); results.append(res)
        el = time.perf_counter() - t0
        print(f"{n + 1}/{len(todo)} {r['id']}: random {res['random']:+.2f} sel@0.5 {res['select_s2']:+.2f} "
              f"steer0.5 {res['steer']['0.5']['final_ir']:+.2f} steer1.0 {res['steer']['1.0']['final_ir']:+.2f} "
              f"oracle {res['oracle16']:+.2f}  ({el / (n + 1):.0f}s/prompt)", flush=True)
    summary, rows_ = summarize(results)
    (DATA / "steer_summary.json").write_text(json.dumps(summary, indent=1))
    figure(rows_, DATA / "figures" / "steering.png", len(results))


if __name__ == "__main__":
    main()
