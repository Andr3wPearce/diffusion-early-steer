"""Figure for one prompt of the rewind experiment: the 12 rewound candidates (columns) at every depth (rows), with final
ImageReward on each image, plus the mean per depth. Usage: python plot_rewind_prompt.py [<id>]  (default: latest done).
Saves data512/figures/rewind_prompt_<id>.png. CPU only."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATA = Path(__file__).parent / "data512"


def main():
    files = sorted((DATA / "rewind").glob("*.json"), key=lambda p: p.stat().st_mtime)
    path = DATA / "rewind" / f"{sys.argv[1]}.json" if len(sys.argv) > 1 else files[-1]
    r = json.loads(path.read_text())
    pid, redo, of = r["id"], r["redo"], np.array(r["orig_final"])
    depths = list(r["depths"])
    rows = [("original", None)] + [(f"step {j}\nσ={r['depths'][j]['sigma']:.2f}", j) for j in depths]
    n = len(redo)
    fig = plt.figure(figsize=(1.45 * n + 3.6, 1.55 * len(rows) + 1.2))
    gs = fig.add_gridspec(len(rows), n + 1, width_ratios=[2.6] + [1] * n, wspace=0.04, hspace=0.3)
    ax = fig.add_subplot(gs[:, 0])
    means = [of[redo].mean()] + [np.mean(r["depths"][j]["new_final"]) for j in depths]
    for row, m in enumerate(means):
        vals = of[redo] if row == 0 else np.array(r["depths"][depths[row - 1]]["new_final"])
        ax.scatter(vals, np.full(n, row) + np.random.default_rng(row).uniform(-0.15, 0.15, n), s=12, alpha=0.6)
        ax.plot([m], [row], "k|", ms=18, mew=2)
    ax.axvline(of.max(), color="tab:green", ls=":", label=f"best of 16 {of.max():+.2f}")
    ax.axvline(of.mean(), color="gray", ls=":", label=f"prompt mean {of.mean():+.2f}")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([lbl for lbl, _ in rows], fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("final ImageReward (| = mean)"); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower left")
    for row, (lbl, j) in enumerate(rows):
        for c, k in enumerate(redo):
            a = fig.add_subplot(gs[row, c + 1]); a.axis("off")
            if j is None:
                img, s = Image.open(DATA / "images" / f"{pid}_seed{k}.png"), of[k]
                col = "black"
            else:
                img, s = Image.open(DATA / "rewind" / f"{pid}_step{j}_seed{k}.png"), r["depths"][j]["new_final"][c]
                col = "tab:green" if s > of[k] else "tab:red"
            a.imshow(img.resize((320, 320)))
            a.set_title(f"{'seed ' + str(k) + ' ' if j is None else ''}{s:+.2f}", fontsize=7, color=col)
    fig.suptitle(f'{pid}: "{r["prompt"][:100]}"\n12 weakest candidates at t=0.5 (columns), rewound to each depth with fresh noise '
                 f'(green = beat its original)', fontsize=10)
    out = DATA / "figures" / f"rewind_prompt_{pid}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
