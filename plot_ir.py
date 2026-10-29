"""Plot ImageReward of the 512px dataset so far: per-category mean (95% CI and +/-1 std) and per-prompt mean +/- std.
Saves data512/figures/imagereward_summary.png. Safe to run while gen512.py is generating."""
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).parent
DATA = HERE / "data512"
CATS = ["coco-style (run.py)", "color", "shape", "texture", "spatial", "3d_spatial", "non_spatial", "numeracy", "complex"]


def main():
    rows = []
    for f in sorted((DATA / "compact").glob("*.pt")):
        d = torch.load(f, weights_only=False)
        rows.append({"split": d["split"], "category": d["category"], "ir": d["ir"].numpy()})

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={"width_ratios": [1, 1.6]})
    for j, (split, color) in enumerate((("eval", "tab:orange"), ("train", "tab:blue"))):
        xs, ms, sds, cis, ns = [], [], [], [], []
        for i, c in enumerate(CATS):
            v = [r["ir"] for r in rows if r["split"] == split and r["category"] == c]
            if not v:
                continue
            v = np.concatenate(v)
            xs.append(i + (j - 0.5) * 0.3); ms.append(v.mean()); sds.append(v.std())
            cis.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0); ns.append(len(v))
        a1.errorbar(xs, ms, yerr=sds, fmt="o", color=color, alpha=0.35, elinewidth=6, capsize=0, label=f"{split}: ±1 std")
        a1.errorbar(xs, ms, yerr=cis, fmt="o", color=color, capsize=5, elinewidth=1.5, label=f"{split}: mean ± 95% CI")
        for x, m, n in zip(xs, ms, ns):
            a1.annotate(str(n), (x, m), textcoords="offset points", xytext=(6, -12), fontsize=7, color=color)
    a1.set_xticks(range(len(CATS))); a1.set_xticklabels([c.split(" ")[0] for c in CATS], rotation=30, ha="right")
    a1.set_ylabel("ImageReward"); a1.set_title("Mean ImageReward by category (numbers = candidates)")
    a1.grid(alpha=0.3, axis="y"); a1.legend(fontsize=8)

    cmap = {c: plt.cm.tab10(i) for i, c in enumerate(CATS)}
    for k, r in enumerate(sorted(rows, key=lambda r: r["ir"].mean())):
        a2.errorbar(k, r["ir"].mean(), yerr=r["ir"].std(), fmt="o" if r["split"] == "train" else "D", ms=3.5,
                    color=cmap[r["category"]], elinewidth=0.8, alpha=0.85)
    for c in CATS:
        a2.plot([], [], "o", color=cmap[c], label=c.split(" ")[0])
    a2.plot([], [], "kD", ms=4, label="eval prompt"); a2.plot([], [], "ko", ms=4, label="train prompt")
    seeds = len(rows[0]["ir"])
    a2.set_xlabel("prompts, sorted by mean ImageReward"); a2.set_ylabel(f"ImageReward (mean ± 1 std over {seeds} seeds)")
    a2.set_title(f"Per-prompt ImageReward ({len(rows)} prompts x {seeds} seeds)")
    a2.grid(alpha=0.3, axis="y"); a2.legend(fontsize=7, ncol=2, loc="lower right")

    all_ir = np.concatenate([r["ir"] for r in rows])
    fig.suptitle(f"SD 3.5 Medium @ 512px: ImageReward over {len(all_ir)} candidates  (overall mean {all_ir.mean():+.2f}, "
                 f"std {all_ir.std():.2f}; mean within-prompt std {np.mean([r['ir'].std() for r in rows]):.2f})", fontsize=12)
    fig.tight_layout()
    out = DATA / "figures" / "imagereward_summary.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110)
    n_train = sum(r["split"] == "train" for r in rows)
    print(f"{len(rows)} prompts ({n_train} train), {len(all_ir)} candidates, mean {all_ir.mean():+.3f} -> {out}")


if __name__ == "__main__":
    main()
