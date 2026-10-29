"""Train our own probe (ProbeSelect's ProbHead) on correctly paired SD3-M features and compare it with the
released checkpoint on the same eval candidates. --data 1024: train_data (8 seeds, the released probe's 80-candidate
test set); --data 512: data512/compact (16 seeds, gen512.py).

Architecture: ProbHead from the ProbeSelect repo with ProbeSelect_SD3.from_pretrained's arguments, except
attention_head_dim 8 instead of 5 (head dim 5 cannot use memory-efficient attention: ~2 GB per image in
training). Loss: pairwise ranking within each prompt's candidates, weighted by the ImageReward gap.
Model selection uses a held-out 10% of the TRAINING prompts; the eval set is only reported, never selected on.
Usage: python train_probe.py --data 512|1024 --row cond|uncond [--epochs N]
"""
import argparse
import json
import random
from collections import defaultdict

import run
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from src.models.components.model import ProbHead

DATASETS = {  # name -> (folder, train glob, eval glob)
    "1024": (run.HERE / "train_data", "train/*.pt", "eval/*.pt"),          # 8 seeds/prompt, released probe's test set
    "512": (run.HERE / "data512" / "compact", "train_*.pt", "eval_*.pt"),  # 16 seeds/prompt, gen512.py
}


def build(head_dim=8):
    return ProbHead(
        latent_channels=48, latent_height=48, num_layers=1,
        latent_width=48, hidden_dims=[128, 256], hidden_dims_mlp=[512], use_batch_norm=True,
        activation="relu", dropout_rate=0.5, resnet_eps=1e-5, resnet_act_fn="silu",
        resnet_groups=16, attention_head_dim=head_dim, output_scale_factor=1.0, downsample_type="conv",
        temb_channels=512, resnet_time_scale_shift="default", num_channels_time=64, text_dim=768, has_text_embed=True,
    )


def load(folder, pattern, row, step=None):
    """step=None: compact files (step 5). step=k: data512/steps cache (extract_steps.py), t = (k+1)/28."""
    groups = []
    for f in sorted(folder.glob(pattern)):
        d = torch.load(f, weights_only=False)
        feat, t = (d[f"feat_{row}"], d["t"]) if step is None else (d[f"feat_{row}"][step], (step + 1) / run.STEPS)
        groups.append({"feat": feat.float().clamp(-500, 500), "ir": d["ir"].float(), "text": d["text_emb"].float(),
                       "t": t, "category": d["category"], "prompt": d["prompt"]})
    return groups


def forward(model, batch, device):
    feat = torch.cat([g["feat"] for g in batch]).to(device)
    n = [len(g["ir"]) for g in batch]
    t = torch.cat([torch.full((k,), g["t"]) for g, k in zip(batch, n)]).to(device)
    text = torch.cat([g["text"].expand(k, -1) for g, k in zip(batch, n)]).to(device)
    if model.training:
        out, _ = model(feat, t, text_emb=text)
    else:  # eval-mode BatchNorm uses running stats, so chunking is exact; keeps head-dim-5 attention in memory
        out = torch.cat([model(feat[i:i + 4], t[i:i + 4], text_emb=text[i:i + 4])[0].reshape(-1) for i in range(0, len(feat), 4)])
    return out.split(n)


def rank_loss(preds, batch, scale=10.0):
    losses = []
    for p, g in zip(preds, batch):
        s = g["ir"].to(p.device)
        dp, ds = (p[:, None] - p[None, :]) * scale, s[:, None] - s[None, :]
        m = ds > 0
        losses.append((F.softplus(-dp[m]) * ds[m]).sum() / ds[m].sum().clamp_min(1e-6))
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model, groups, device, bs=8):
    model.eval()
    rhos, hits, pick, rand, oracle = [], [], [], [], []
    for i in range(0, len(groups), bs):
        b = groups[i:i + bs]
        for p, g in zip(forward(model, b, device), b):
            p, s = p.float().cpu().numpy(), g["ir"].numpy()
            rhos.append(spearmanr(p, s).statistic); hits.append(int(p.argmax() == s.argmax()))
            pick.append(s[p.argmax()]); rand.append(s.mean()); oracle.append(s.max())
    r = {"mean_rho": float(np.nanmean(rhos)), "top1_hits": int(sum(hits)), "n": len(groups),
         "mean_pick_ir": float(np.mean(pick)), "mean_random_ir": float(np.mean(rand)), "mean_oracle_ir": float(np.mean(oracle))}
    r["gain_fraction"] = (r["mean_pick_ir"] - r["mean_random_ir"]) / max(r["mean_oracle_ir"] - r["mean_random_ir"], 1e-9)
    r["rhos"] = [float(x) for x in rhos]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=sorted(DATASETS), default="512")
    ap.add_argument("--row", choices=["cond", "uncond"], default="cond")
    ap.add_argument("--step", type=int, default=None, help="use data512/steps cache at this step (extract_steps.py)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--groups-per-batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)
    device = "cuda"

    folder, train_glob, eval_glob = DATASETS[args.data]
    if args.step is not None:
        assert args.data == "512", "--step needs the 512 steps cache"
        folder = run.HERE / "data512" / "steps"
    groups = load(folder, train_glob, args.row, args.step)
    by_cat = defaultdict(list)
    for g in groups:
        by_cat[g["category"]].append(g)
    rng = random.Random(args.seed)
    train, holdout = [], []
    for cat in sorted(by_cat):
        items = by_cat[cat][:]; rng.shuffle(items)
        k = max(1, round(0.1 * len(items)))
        holdout += items[:k]; train += items[k:]
    evalset = load(folder, eval_glob, args.row, args.step)
    n_cand = len(evalset[0]["ir"])
    print(f"data={args.data} row={args.row}: {len(train)} train / {len(holdout)} holdout prompts, eval {len(evalset)} prompts "
          f"(x{n_cand} candidates)", flush=True)

    # Reference: the released checkpoint as ProbeSelect_SD3 runs it (uncond row, head dim 5) on the same eval set.
    released = run.build_probe(device)
    released_eval = evaluate(released, load(folder, eval_glob, "uncond", args.step), device)
    del released

    model = build().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    out_dir = run.HERE / "runs" / f"probe{args.data}_{args.row}_step{5 if args.step is None else args.step}_n{len(train)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best, history = None, []
    for epoch in range(1, args.epochs + 1):
        model.train(); rng.shuffle(train); tot = 0.0
        for i in range(0, len(train), args.groups_per_batch):
            b = train[i:i + args.groups_per_batch]
            if len(b) < 2:
                continue  # BatchNorm needs more than one prompt per batch
            loss = rank_loss(forward(model, b, device), b)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        sched.step()
        h, e = evaluate(model, holdout, device), evaluate(model, evalset, device)
        history.append({"epoch": epoch, "loss": tot / len(train), "holdout": {k: v for k, v in h.items() if k != "rhos"},
                        "eval": {k: v for k, v in e.items() if k != "rhos"}})
        print(f"ep {epoch:3d} loss {tot / len(train):.4f} | holdout rho {h['mean_rho']:+.3f} top1 {h['top1_hits']}/{h['n']} "
              f"gain {h['gain_fraction']:+.0%} | eval rho {e['mean_rho']:+.3f} top1 {e['top1_hits']}/{e['n']} gain {e['gain_fraction']:+.0%}", flush=True)
        if best is None or h["mean_rho"] > best["holdout"]["mean_rho"]:
            best = {"epoch": epoch, "holdout": h, "eval": e}
            torch.save(model.state_dict(), out_dir / "best.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    best["released_checkpoint_eval"] = released_eval
    (out_dir / "best.json").write_text(json.dumps(best, indent=1))
    e, r = best["eval"], released_eval
    print(f"\nbest epoch {best['epoch']} (chosen on holdout rho {best['holdout']['mean_rho']:+.3f})")
    print(f"EVAL ({e['n']} prompts x {n_cand}): ours rho {e['mean_rho']:+.3f}, top1 {e['top1_hits']}/{e['n']}, pick IR {e['mean_pick_ir']:+.3f} "
          f"(random {e['mean_random_ir']:+.3f}, oracle {e['mean_oracle_ir']:+.3f}, gain {e['gain_fraction']:+.0%})")
    print(f"      released checkpoint: rho {r['mean_rho']:+.3f}, top1 {r['top1_hits']}/{r['n']}, pick IR {r['mean_pick_ir']:+.3f}, "
          f"gain {r['gain_fraction']:+.0%}" + ("  (trained at 1024px; 512 is out of distribution)" if args.data == "512" else ""))
    print(f"      chance: top1 {e['n'] / n_cand:.1f}/{e['n']}")


if __name__ == "__main__":
    main()
