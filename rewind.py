"""Rewind experiment: judge at step J (t = 0.5), keep the top K, send the rest back to an early step with fresh noise.

Per prompt (16 saved candidates): x0_hat ImageReward at step J ranks them. The bottom 16-K are rewound: from their
step-J state (exact velocity from the saved trajectory) x0_hat = x - s*v; re-noised to step j as
x' = (1 - sigma_j) * x0_hat + sigma_j * xi (j = 0: pure noise, i.e. a fresh seed), denoised through step J-1, judged at
step J again (x0_hat ImageReward), and run to the end and scored, so the true effect is measured, not just predicted.
For each depth j: effect on the rewound candidates (final IR after vs before), pool ceiling (best final of the K kept +
rewound), and a practical policy (pick by step-J score from the pool, finish it) vs selecting from the 16 originals.
Outputs data512/rewind/<id>.json, rewound finals as data512/rewind/<id>_step<j>_seed<k>.png, data512/rewind_summary.json,
data512/figures/rewind.png.
"""
import json
import time

import run
import numpy as np
import torch

from steer import DATA, Sampler, euler, load_embeds

J = 13                          # judge at step 13 (x0_hat at t = 14/28 = 0.5)
K = 4                           # keep the top K, rewind the other 16 - K
DEPTHS = [8, 6, 4, 3, 2, 1, 0]  # rewind targets (step index); 0 = pure noise


@torch.no_grad()
def rewind_prompt(sm, idx, r):
    d = torch.load(DATA / "full" / f"{r['id']}.pt", weights_only=False, mmap=True)
    xh = torch.load(DATA / "xhat0" / f"{r['id']}.pt", weights_only=False)
    sig, ts, lat = d["sigmas"].float(), d["timesteps"].float(), d["latents"]
    final = d["ir"].numpy().astype(np.float64)
    judge = xh["score"][xh["steps"].index(J)].numpy().astype(np.float64)
    emb = load_embeds(idx, sm.device)
    order = np.argsort(-judge)
    keep, redo = order[:K], order[K:]
    x = lat[redo, J].to(sm.device)
    v = (lat[redo, J + 1].to(sm.device).float() - x.float()) / (sig[J + 1] - sig[J])
    x0_hat = x.float() - sig[J] * v
    out = {"id": r["id"], "prompt": r["prompt"], "category": r["category"], "keep": keep.tolist(), "redo": redo.tolist(),
           "orig_final": final.tolist(), "orig_judge": judge.tolist(), "depths": {}}
    for j in DEPTHS:
        g = torch.Generator("cpu").manual_seed(30_000 + 100 * idx + j)
        xi = torch.randn(tuple(x0_hat.shape), generator=g).to(sm.device)
        xb = ((1 - sig[j]) * x0_hat + sig[j] * xi).to(torch.bfloat16)
        xb, _ = sm.run_steps(xb, j, J, sig, ts, emb)                    # back to the judging point
        vj = sm.velocity(xb, ts[J].to(sm.device), emb)
        new_judge, _ = sm.score((xb.float() - sig[J] * vj.float()).to(xb.dtype), r["prompt"])
        xb = euler(xb, vj, sig, J)
        xb, _ = sm.run_steps(xb, J + 1, run.STEPS, sig, ts, emb)       # finish all of them
        new_final, pils = sm.score(xb, r["prompt"])
        for k, p in zip(redo, pils):
            p.save(DATA / "rewind" / f"{r['id']}_step{j}_seed{int(k)}.png")
        out["depths"][str(j)] = {"sigma": float(sig[j]), "new_judge": new_judge.tolist(), "new_final": new_final.tolist()}
    return out


def analyze(results):
    n = len(results)
    S = (J + 1) / 28
    base_cost = 16 * S + (1 - S)  # 16 candidates to the judging point, finish one
    rows, table = {}, {}
    sel = np.array([r["orig_final"][int(np.argmax(r["orig_judge"]))] for r in results])
    rows["select best of 16 @t=0.5"] = (sel, base_cost)
    rows["random single sample"] = (np.array([np.mean(r["orig_final"]) for r in results]), 1.0)
    rows["oracle best of 16 originals"] = (np.array([max(r["orig_final"]) for r in results]), 16.0)
    print(f"\n{n} prompts; rewound {16 - K} of 16 at step {J} (t={S:.2f}); select-at-0.5 baseline {sel.mean():+.3f}")
    print("depth sigma | rewound cands: final before -> after (improved %) | pool ceiling | policy (pick by step-13 score) vs select@0.5 | compute")
    for j in DEPTHS:
        before, after, ceil, pol = [], [], [], []
        for r in results:
            dd = r["depths"][str(j)]
            of, nf = np.array(r["orig_final"]), np.array(dd["new_final"])
            before += of[r["redo"]].tolist(); after += nf.tolist()
            pool_final = np.concatenate([of[r["keep"]], nf])
            pool_judge = np.concatenate([np.array(r["orig_judge"])[r["keep"]], np.array(dd["new_judge"])])
            ceil.append(pool_final.max()); pol.append(pool_final[int(np.argmax(pool_judge))])
        before, after, ceil, pol = map(np.array, (before, after, ceil, pol))
        cost = base_cost + (16 - K) * (J + 1 - j) / 28
        rows[f"rewind to step {j}: policy"] = (pol, cost)
        d = pol - sel
        table[j] = {"sigma": results[0]["depths"][str(j)]["sigma"], "rewound_before": float(before.mean()), "rewound_after": float(after.mean()),
                    "rewound_improved_frac": float((after > before).mean()), "pool_ceiling": float(ceil.mean()),
                    "policy": float(pol.mean()), "policy_vs_select": float(d.mean()), "policy_vs_select_se": float(d.std(ddof=1) / np.sqrt(n)),
                    "compute": cost}
        t = table[j]
        print(f"  {j:2d}  {t['sigma']:.2f} | {t['rewound_before']:+.3f} -> {t['rewound_after']:+.3f} ({t['rewound_improved_frac']:.0%}) | "
              f"{t['pool_ceiling']:+.3f} | {t['policy']:+.3f} ({t['policy_vs_select']:+.3f} ± {t['policy_vs_select_se']:.3f}) | {cost:.1f}")
    return table, rows


def figure(table, rows, path, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sig = [table[j]["sigma"] for j in DEPTHS]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.5))
    a1.plot(sig, [table[j]["rewound_before"] for j in DEPTHS], "s--", color="gray", label="rewound candidates: before (original final)")
    a1.plot(sig, [table[j]["rewound_after"] for j in DEPTHS], "o-", label="rewound candidates: after")
    a1.plot(sig, [table[j]["pool_ceiling"] for j in DEPTHS], "^-", label="pool ceiling (best final of kept + rewound)")
    a1.plot(sig, [table[j]["policy"] for j in DEPTHS], "D-", label="policy: pick by step-13 score, finish")
    a1.axhline(rows["select best of 16 @t=0.5"][0].mean(), color="tab:green", ls=":", label="select best of 16 @t=0.5")
    a1.axhline(rows["oracle best of 16 originals"][0].mean(), color="tab:brown", ls=":", label="oracle best of 16 originals")
    for j, s in zip(DEPTHS, sig):
        a1.annotate(f"step {j}", (s, table[j]["rewound_after"]), textcoords="offset points", xytext=(0, -14), fontsize=7, ha="center")
    a1.set_xlabel("noise level rewound to (sigma; 1.0 = pure noise)"); a1.set_ylabel(f"final ImageReward (mean over {n} prompts)")
    a1.set_title("How much does rewinding change the final image reward?"); a1.grid(alpha=0.3); a1.legend(fontsize=7)
    for name, (v, cost) in rows.items():
        a2.errorbar(cost, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), fmt="o", capsize=3, label=name)
    a2.set_xlabel("compute (full generations)"); a2.set_ylabel("final ImageReward (mean ± se)")
    a2.set_title("Compute vs result"); a2.grid(alpha=0.3); a2.legend(fontsize=7)
    fig.suptitle(f"Rewind poor candidates (bottom {16 - K} of 16 at t=0.5) to an earlier noise level, SD 3.5 Medium @512")
    fig.tight_layout(); fig.savefig(path, dpi=110)


def main():
    rows = json.loads((DATA / "prompts.json").read_text())
    (DATA / "rewind").mkdir(exist_ok=True)
    todo = [(i, r) for i, r in enumerate(rows) if r["split"] == "eval" and (DATA / "xhat0" / f"{r['id']}.pt").exists()]
    sm = Sampler()
    results, t0, done_now = [], time.perf_counter(), 0
    for n, (i, r) in enumerate(todo):
        path = DATA / "rewind" / f"{r['id']}.json"
        if path.exists():
            results.append(json.loads(path.read_text())); continue
        res = rewind_prompt(sm, i, r)
        path.write_text(json.dumps(res, indent=1)); results.append(res); done_now += 1
        el = (time.perf_counter() - t0) / done_now
        of = np.array(res["orig_final"])[res["redo"]].mean()
        print(f"{n + 1}/{len(todo)} {r['id']}: rewound before {of:+.2f} | after " +
              " ".join(f"s{j}:{np.mean(res['depths'][str(j)]['new_final']):+.2f}" for j in DEPTHS) +
              f"  ({el:.0f}s/prompt, ~{el * (len(todo) - n - 1) / 60:.0f} min left)", flush=True)
    table, rows_ = analyze(results)
    (DATA / "rewind_summary.json").write_text(json.dumps({str(k): v for k, v in table.items()}, indent=1))
    figure(table, rows_, DATA / "figures" / "rewind.png", len(results))


if __name__ == "__main__":
    main()
