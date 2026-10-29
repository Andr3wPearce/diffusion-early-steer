"""Write data512/status.html: a self-refreshing local progress page for the probe experiments.
Run once, or loop it: `bash status_loop.sh` (rewrites every 60 s). Open data512/status.html in a browser."""
import html
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data512"
TOTAL_TRAIN = 5559


def tail_match(path, pattern, n=1):
    try:
        lines = [l for l in path.read_text(encoding="utf-8", errors="ignore").splitlines() if re.search(pattern, l)]
        return lines[-n:]
    except FileNotFoundError:
        return []


def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
        used, total, util, temp = [x.strip() for x in out.split(",")]
        return int(used), int(total), int(util), int(temp)
    except Exception:
        return None


def bar(frac, label):
    frac = max(0.0, min(1.0, frac))
    return f'<div class="bar"><div class="fill" style="width:{frac * 100:.1f}%"></div><span>{html.escape(label)}</span></div>'


def disk_gb(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def main():
    now = time.time()
    compact = list((DATA / "compact").glob("*.pt"))
    n_train = sum(f.name.startswith("train_") for f in compact)
    n_eval = sum(f.name.startswith("eval_") for f in compact)
    recent = [f for f in compact if now - f.stat().st_mtime < 1800]
    rate = len(recent) * 2  # prompts per hour over the last 30 min
    gen_lines = tail_match(HERE / "gen512.log", r"^(eval|train)_\d+ ", 6)
    last_age = now - max((f.stat().st_mtime for f in compact), default=now)
    gen_alive = last_age < 600

    jobs = []
    for log, name, pat in ((HERE / "xhat0_features.log", "x0_hat ImageReward features", r"prompts|DONE|Traceback"),
                           (HERE / "combo.log", "early-selection comparison", r"^step|Traceback|^\d+ train"),
                           (HERE / "train512_cond.log", "neural probe training", r"^ep|^EVAL|Traceback"),
                           (HERE / "steer.log", "steering experiment", r"^\d+/\d+ |Traceback|prompts\. Final"),
                           (HERE / "rewind.log", "rewind experiment", r"^\d+/\d+ |Traceback|prompts; rewound")):
        if log.exists() and now - log.stat().st_mtime < 6 * 3600:
            last = tail_match(log, pat)
            m = re.match(r"(\d+)/(\d+)[ ]", last[0]) if last else None
            frac = int(m.group(1)) / int(m.group(2)) if m else (1.0 if last and "DONE" in last[0] else None)
            active = now - log.stat().st_mtime < 300
            jobs.append((name, last[0] if last else "", frac, active))

    g = gpu()
    figs = [p for p in ("figures/rewind.png", "figures/steering.png", "figures/early_selection_curve.png", "figures/imagereward_summary.png", "figures/xhat0_eval_0000.png")
            if (DATA / p).exists()]
    thumbs = sorted((DATA / "images").glob("train_*_seed*.png"), key=lambda p: p.stat().st_mtime)[-8:]

    early = ""
    ej = DATA / "early_selection.json"
    if ej.exists():
        res = json.loads(ej.read_text())
        steps = sorted({int(k.split("_")[0][4:]) for k in res})
        methods = sorted({k.split("_")[1] for k in res})
        rows = "".join(f"<tr><td>{s} (t={(s + 1) / 28:.2f})</td>" + "".join(
            f"<td>{res[f'step{s}_{m}']['cv_train']['rho']:+.2f} / {res[f'step{s}_{m}']['eval']['rho']:+.2f}</td>" for m in methods) + "</tr>"
            for s in steps)
        early = (f"<h2>Early-selection results <small>(within-prompt Spearman rho: out-of-fold train / eval)</small></h2>"
                 f"<table><tr><th>step</th>{''.join(f'<th>{m}</th>' for m in methods)}</tr>{rows}</table>"
                 "<p class='note'>A: x0_hat ImageReward score · B: PCA-stats probe · C: x0_hat IR embedding · D: A+B · E: C+B · F: A+C</p>")

    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="30">
<title>Probe Experiment Status</title><style>
:root {{ --bg:#f7f7f5; --fg:#1d1d1b; --muted:#6b6b66; --card:#fff; --line:#e3e3df; --accent:#3b6fd8; --ok:#2f9e5b; --warn:#c77d16; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#18181a; --fg:#ececea; --muted:#9a9a95; --card:#222225; --line:#34343a; --accent:#6b9bff; --ok:#4cc27c; --warn:#e3a33b; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0; padding:24px 16px; }}
main {{ max-width:1100px; margin:0 auto; }} h1 {{ font-size:22px; margin:0 0 4px; }} h2 {{ font-size:16px; margin:28px 0 10px; }}
small, .note {{ color:var(--muted); font-weight:normal; font-size:13px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:12px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
.big {{ font-size:26px; font-weight:600; }} .dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }}
.bar {{ position:relative; height:22px; background:var(--line); border-radius:6px; overflow:hidden; margin:6px 0; }}
.fill {{ position:absolute; inset:0 auto 0 0; background:var(--accent); opacity:.85; }}
.bar span {{ position:relative; padding-left:8px; font-size:13px; line-height:22px; }}
pre {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px; overflow-x:auto; font-size:12.5px; margin:0; }}
table {{ border-collapse:collapse; font-size:13px; width:100%; overflow-x:auto; display:block; }} td, th {{ border:1px solid var(--line); padding:4px 8px; text-align:center; }}
img.fig {{ width:100%; border:1px solid var(--line); border-radius:8px; margin-bottom:12px; background:#fff; }}
.thumbs {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(120px, 1fr)); gap:8px; }} .thumbs img {{ width:100%; border-radius:6px; }}
</style></head><body><main>
<h1>Probe experiment status</h1><small>updated {datetime.now():%H:%M:%S} · page reloads every 30 s</small>
<h2>Data generation (SD 3.5 Medium @ 512px, 16 seeds/prompt)</h2>
<div class="cards">
 <div class="card"><span class="dot" style="background:{'var(--ok)' if gen_alive else 'var(--warn)'}"></span>{'running' if gen_alive else f'no new prompt for {last_age / 60:.0f} min'}
  <div class="big">{n_train} <small>train prompts</small></div><small>{n_train * 16:,} training images · eval {n_eval}/25 prompts</small></div>
 <div class="card"><div class="big">{rate} <small>prompts/hour</small></div><small>over the last 30 min{f' · ~{(500 - n_train) / rate:.1f} h to 500' if rate and n_train < 500 else ''}</small></div>
 <div class="card"><div class="big">{disk_gb(DATA):.0f} <small>GB on disk</small></div><small>data512/</small></div>
 <div class="card">{f'<div class="big">{g[0] / 1024:.1f} <small>/ {g[1] / 1024:.0f} GB GPU</small></div><small>utilization {g[2]}% · {g[3]}°C</small>' if g else 'GPU: n/a'}</div>
</div>
{bar(n_train / TOTAL_TRAIN, f'{n_train} / {TOTAL_TRAIN} train prompts available')}
<pre>{html.escape(chr(10).join(gen_lines))}</pre>
<h2>Analysis jobs</h2>
{''.join(f'<div class="card" style="margin-bottom:8px"><b>{html.escape(n)}</b> <small>{"active" if a else "idle/finished"}</small>' + (bar(fr, f"{fr:.0%}") if fr is not None else '') + f'<pre>{html.escape(l)}</pre></div>' for n, l, fr, a in jobs) or '<p class="note">none in the last 6 h</p>'}
{early}
<h2>Figures</h2>{''.join(f'<img class="fig" src="{p}?t={int(now)}" alt="{p}">' for p in figs)}
<h2>Latest generated images</h2><div class="thumbs">{''.join(f'<img src="images/{p.name}" title="{p.name}">' for p in thumbs)}</div>
</main></body></html>"""
    (DATA / "status.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
