"""Early selection: how well can we predict final ImageReward ranking at each timestep, and from what?

Per step (xhat0_features.STEPS), within-prompt ridge rankers on:
  A  x0_hat ImageReward score (no fitting)
  B  PCA-stats probe: sign-invariant summaries of block-20 PCA features, cond row (see simple_probe.py)
  C  x0_hat ImageReward 768-d joint embedding
  D  A + B        E  C + B        F  A + C
Features and targets are centered within each prompt. Lambda by 5-fold CV over training prompts. Reports eval (25
prompts) and out-of-fold CV over training prompts (every training prompt scored by a model that never saw it).
Saves data512/early_selection.json and data512/figures/early_selection_curve.png.
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from simple_probe import featurize

DATA = Path(__file__).parent / "data512"
LAMBDAS = np.logspace(-1, 5, 13)
METHODS = {"A": ["score"], "B": ["pca"], "C": ["emb"], "D": ["score", "pca"], "E": ["emb", "pca"], "F": ["score", "emb"]}
NAMES = {"A": "x0_hat IR score (no fit)", "B": "PCA-stats probe", "C": "x0_hat IR embedding", "D": "A + B", "E": "C + B", "F": "A + C"}


def load_all():
    prompts = []
    for f in sorted((DATA / "xhat0").glob("*.pt")):
        x = torch.load(f, weights_only=False)
        full = torch.load(DATA / "full" / f.name, weights_only=False, mmap=True)
        pca = {s: featurize(full["feat_cond"][:, s]) for s in x["steps"]}
        prompts.append({"split": x["split"], "ir": x["ir"].numpy().astype(np.float64), "steps": x["steps"],
                        "score": x["score"].numpy().astype(np.float64), "emb": x["emb"].float().numpy(), "pca": pca})
    return prompts


def design(p, si, step, parts):
    cols = []
    for part in parts:
        if part == "score":
            cols.append(p["score"][si][:, None])
        elif part == "emb":
            cols.append(p["emb"][si])
        else:
            cols.append(p["pca"][step])
    X = np.concatenate(cols, 1)
    return X - X.mean(0), p["ir"] - p["ir"].mean(), p["ir"]


def ridge(groups, lam):
    X = np.concatenate([g[0] for g in groups]); y = np.concatenate([g[1] for g in groups])
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd
    w = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ y)
    return lambda Xc: ((Xc - mu) / sd) @ w


def metrics(preds_and_y):
    rhos = [spearmanr(p, y).statistic for p, y in preds_and_y]
    pick = np.mean([y[np.argmax(p)] for p, y in preds_and_y]); rand = np.mean([y.mean() for _, y in preds_and_y])
    orc = np.mean([y.max() for _, y in preds_and_y])
    return {"rho": float(np.nanmean(rhos)), "rho_se": float(np.nanstd(rhos) / np.sqrt(len(rhos))),
            "top1": int(sum(np.argmax(p) == np.argmax(y) for p, y in preds_and_y)), "n": len(rhos),
            "gain": float((pick - rand) / (orc - rand))}


def main():
    prompts = load_all()
    train = [p for p in prompts if p["split"] == "train"]; evals = [p for p in prompts if p["split"] == "eval"]
    steps = prompts[0]["steps"]
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(len(train)), 5)
    print(f"{len(train)} train / {len(evals)} eval prompts x 16; chance top1: eval {len(evals) / 16:.1f}, train {len(train) / 16:.1f}")
    results = {}
    for si, step in enumerate(steps):
        t = (step + 1) / 28
        for m, parts in METHODS.items():
            tr = [design(p, si, step, parts) for p in train]; ev = [design(p, si, step, parts) for p in evals]
            if m == "A":  # no fitting: the score itself
                oof = [(g[0][:, 0], g[2]) for g in tr]; ev_pred = [(g[0][:, 0], g[2]) for g in ev]; lam = None
            else:
                best = None
                for lam_c in LAMBDAS:
                    oof_c = [None] * len(tr)
                    for k in range(5):
                        f = ridge([tr[i] for j in range(5) if j != k for i in folds[j]], lam_c)
                        for i in folds[k]:
                            oof_c[i] = (f(tr[i][0]), tr[i][2])
                    r = metrics(oof_c)["rho"]
                    if best is None or r > best[0]:
                        best = (r, lam_c, oof_c)
                _, lam, oof = best
                f = ridge(tr, lam)
                ev_pred = [(f(g[0]), g[2]) for g in ev]
            results[(step, m)] = {"t": t, "lambda": lam, "cv_train": metrics(oof), "eval": metrics(ev_pred)}
        line = "  ".join(f"{m}:{results[(step, m)]['cv_train']['rho']:+.2f}/{results[(step, m)]['eval']['rho']:+.2f}" for m in METHODS)
        print(f"step {step:2d} t={t:.2f}  (cv_train/eval rho)  {line}", flush=True)

    (DATA / "early_selection.json").write_text(json.dumps({f"step{s}_{m}": v for (s, m), v in results.items()}, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = [(s + 1) / 28 for s in steps]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for m in METHODS:
        for ax, key, label in ((axes[0], "cv_train", f"out-of-fold over {len(train)} train prompts"), (axes[1], "eval", f"{len(evals)} eval prompts")):
            r = np.array([results[(s, m)][key]["rho"] for s in steps]); se = np.array([results[(s, m)][key]["rho_se"] for s in steps])
            ax.plot(ts, r, "o-", label=f"{m}: {NAMES[m]}", lw=2 if m in ("A", "E", "F") else 1)
            ax.fill_between(ts, r - se, r + se, alpha=0.12)
            ax.set_title(f"Within-prompt Spearman rho vs final ImageReward ({label})")
    for ax in axes:
        ax.axhline(0, color="k", lw=0.8); ax.axvline(0.214, color="gray", ls="--", lw=1)
        ax.set_xlabel("denoising progress t at which we predict"); ax.set_ylabel("mean Spearman rho (±1 se)"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Early selection on SD 3.5 Medium @512px, 16 candidates/prompt (dashed: ProbeSelect's t≈0.21)")
    fig.tight_layout(); fig.savefig(DATA / "figures" / "early_selection_curve.png", dpi=110)


if __name__ == "__main__":
    main()
