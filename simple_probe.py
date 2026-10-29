"""Simple within-prompt ranker on sign-invariant summaries of the block-20 PCA features (data512/steps cache).

ProbeSelect's PCA is per sample, so channel k is a different (arbitrarily signed) direction for every candidate.
These features don't depend on that: per-component log energy (the PCA spectrum, 48) and a 4x4 spatial map of |x|
for the top 8 components (128). Features and ImageReward are centered within each prompt, so only within-prompt
differences are learned. Ridge regression; lambda chosen by 5-fold cross-validation over training prompts.
Reports eval mean Spearman rho / top-1 hits / gain for each (step, row). CPU only.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

DATA = Path(__file__).parent / "data512" / "steps"
STEPS = [5, 13, 20, 27]
LAMBDAS = np.logspace(-2, 4, 13)


def featurize(x):  # x: [n, 48, 48, 48]
    x = x.float()
    energy = torch.log(x.pow(2).mean(dim=(2, 3)) + 1e-6)                          # [n, 48]
    spatial = F.adaptive_avg_pool2d(x[:, :8].abs(), 4).flatten(1)                  # [n, 128]
    return torch.cat([energy, torch.log(spatial + 1e-6)], 1).numpy()


def load(split, step, row):
    groups = []
    for f in sorted(DATA.glob(f"{split}_*.pt")):
        d = torch.load(f, weights_only=False)
        X, y = featurize(d[f"feat_{row}"][step]), d["ir"].numpy().astype(np.float64)
        groups.append((X - X.mean(0), y - y.mean(), y))
    return groups


def fit(groups, lam, mu=None, sd=None):
    X = np.concatenate([g[0] for g in groups]); y = np.concatenate([g[1] for g in groups])
    if mu is None:
        mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd
    w = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ y)
    return w, mu, sd


def score(groups, w, mu, sd):
    rhos, hits, pick, rand, orc = [], [], [], [], []
    for Xc, yc, y in groups:
        p = ((Xc - mu) / sd) @ w
        rhos.append(spearmanr(p, y).statistic); hits.append(int(p.argmax() == y.argmax()))
        pick.append(y[p.argmax()]); rand.append(y.mean()); orc.append(y.max())
    gain = (np.mean(pick) - np.mean(rand)) / (np.mean(orc) - np.mean(rand))
    return float(np.nanmean(rhos)), int(sum(hits)), float(gain)


def main():
    rng = np.random.default_rng(0)
    n_train = len(list(DATA.glob("train_*.pt")))
    print(f"{n_train} train prompts, {len(list(DATA.glob('eval_*.pt')))} eval prompts (x16); chance top1 = {len(list(DATA.glob('eval_*.pt'))) / 16:.1f}")
    print("step  t     row     lambda   cv_rho   eval_rho  top1  gain")
    for step in STEPS:
        for row in ("cond", "uncond"):
            train, evalset = load("train", step, row), load("eval", step, row)
            folds = np.array_split(rng.permutation(len(train)), 5)
            cv = []
            for lam in LAMBDAS:
                r = []
                for k in range(5):
                    tr = [train[i] for j in range(5) if j != k for i in folds[j]]
                    w, mu, sd = fit(tr, lam)
                    r.append(score([train[i] for i in folds[k]], w, mu, sd)[0])
                cv.append(np.mean(r))
            lam = LAMBDAS[int(np.argmax(cv))]
            w, mu, sd = fit(train, lam)
            rho, hits, gain = score(evalset, w, mu, sd)
            print(f"{step:4d}  {(step + 1) / 28:.2f}  {row:6s}  {lam:8.2f}  {max(cv):+.3f}    {rho:+.3f}   {hits:2d}   {gain:+.0%}")


if __name__ == "__main__":
    main()
