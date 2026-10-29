"""Cache a few timesteps from data512/full/*.pt so experiments don't reread 280 MB per prompt.
Writes data512/steps/<id>.pt with feat_{uncond,cond}[step] [16, 48, 48, 48] fp16 for STEPS, plus ir, text_emb,
prompt, category, split. Skips prompts already cached; safe to run while gen512.py is generating."""
import time
from pathlib import Path

import torch

DATA = Path(__file__).parent / "data512"
STEPS = [5, 13, 20, 27]  # t = (i+1)/28 = 0.21, 0.50, 0.75, 1.00


def main():
    out = DATA / "steps"
    out.mkdir(exist_ok=True)
    t0, n = time.perf_counter(), 0
    for f in sorted((DATA / "full").glob("*.pt")):
        dest = out / f.name
        if dest.exists():
            continue
        d = torch.load(f, weights_only=False, mmap=True)
        rec = {k: d[k] for k in ("ir", "text_emb", "prompt", "category", "split")}
        for row in ("uncond", "cond"):
            rec[f"feat_{row}"] = {s: d[f"feat_{row}"][:, s].clone() for s in STEPS}
        tmp = dest.with_suffix(".tmp")
        torch.save(rec, tmp)
        tmp.replace(dest)
        n += 1
    print(f"cached {n} new prompts ({len(list(out.glob('*.pt')))} total) in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
