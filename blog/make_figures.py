"""Figures for the Prophet vs Timestral post. Run: .venv\\Scripts\\python.exe blog\\make_figures.py"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gmean

sys.path.insert(0, str(Path(__file__).parent))
from analysis import leaderboard, load  # noqa: E402
from style import BG, GRID, INK, INK2, OTHER, PROPHET, TIMESTRAL  # noqa: E402

OUT = Path(__file__).parent

fev_all, combined, names = load()
N = len(names)


def color_for(m):
    return TIMESTRAL if m.startswith("Timestral") else PROPHET if m.startswith("Prophet") else OTHER


# 1. Leaderboard: skill score = % less error than Seasonal Naive (SQL, geometric mean over tasks)
lb = leaderboard(combined, "SQL")
models = ["Timestral-209M", "timesfm-3", "Chronos-2", "Stat. Ensemble", "LightGBM", "AutoARIMA",
          "Prophet (+cov, >=0)", "Prophet (default)"]
labels = {"Prophet (+cov, >=0)": "Prophet (+covariates)", "Stat. Ensemble": "Stat. ensemble (ETS/ARIMA/Theta)"}
sk = lb.loc[models, "skill_score"] * 100

fig, ax = plt.subplots(figsize=(10, 6.2))
y = np.arange(len(models))[::-1]
ax.barh(y, sk, height=0.62, color=[color_for(m) for m in models], edgecolor=BG, linewidth=2)
for yi, v in zip(y, sk):
    ax.text(v + 0.8, yi, f"{v:.0f}%", va="center", fontsize=13, color=INK, fontweight="bold")
ax.set_yticks(y, [labels.get(m, m) for m in models], fontsize=13)
for t, m in zip(ax.get_yticklabels(), models):
    t.set_color(INK if m.startswith(("Timestral", "Prophet")) else INK2)
    t.set_fontweight("bold" if m.startswith(("Timestral", "Prophet")) else "normal")
ax.set_xlim(0, 58)
ax.xaxis.grid(True, color=GRID, lw=1)
ax.set_axisbelow(True)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("% error reduction vs. Seasonal Naive  (SQL skill score, higher is better)", fontsize=12)
fig.suptitle("Timestral cuts forecast error ~5× more than Prophet", x=0.02, ha="left", fontsize=19,
             fontweight="bold", y=0.975)
fig.text(0.02, 0.885, f"fev-bench, {N} tasks (energy, retail, health, macro, weather…) · probabilistic forecasts",
         fontsize=12.5, color=INK2)
fig.text(0.02, 0.015, "Prophet: my runs, out of the box. Other models: published fev-bench results.",
         fontsize=10.5, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 0.87))
fig.savefig(OUT / "fig1_leaderboard.png")

# 2. Task by task: Timestral-209M vs Prophet (+cov), each relative to Seasonal Naive
pv = combined.pivot_table(index="task_name", columns="model_name", values="SQL")
rel = pv.div(pv["Seasonal Naive"], axis=0)
P, T = "Prophet (+cov, >=0)", "Timestral-209M"
d = rel[[P, T]].dropna()
wins = int((d[T] < d[P]).sum())
ratio = gmean(d[T] / d[P])

fig, ax = plt.subplots(figsize=(8.4, 8.4))
lo, hi = 0.2, 2.5
ax.fill_between([lo, hi], [lo, hi], lo, color=TIMESTRAL, alpha=0.06, lw=0)
ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--")
ax.axvline(1, color=GRID, lw=1.5, zorder=0)
ax.axhline(1, color=GRID, lw=1.5, zorder=0)
ax.scatter(d[P], d[T], s=70, color=TIMESTRAL, edgecolor=BG, linewidth=2, zorder=3)
ax.set_xscale("log"), ax.set_yscale("log")
ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)
ticks = [0.25, 0.5, 1, 2]
ax.set_xticks(ticks, [f"{t:g}" for t in ticks]), ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
ax.minorticks_off()
ax.set_xlabel("Prophet error  (÷ Seasonal Naive)", fontsize=13)
ax.set_ylabel("Timestral-209M error  (÷ Seasonal Naive)", fontsize=13)
ax.text(1.6, 0.23, "Timestral better ↘", color=TIMESTRAL, fontsize=13, fontweight="bold", ha="center")
ax.text(0.32, 1.9, "↖ Prophet better", color=PROPHET, fontsize=13, fontweight="bold", ha="center")
ax.text(0.205, 1.03, "Seasonal Naive level", color=INK2, fontsize=10, va="bottom")
fig.suptitle(f"Timestral beats Prophet on {wins} of {len(d)} tasks", x=0.03, ha="left", fontsize=19,
             fontweight="bold", y=0.975)
fig.text(0.03, 0.915, f"Each dot = one fev-bench task. Typical error: {100 * (1 - ratio):.0f}% lower than Prophet "
         f"(geometric mean).", fontsize=12, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.savefig(OUT / "fig2_task_by_task.png")

beats = {m: int((rel[m].dropna() < 1).sum()) for m in [T, P, "Prophet (default)"]}
print(f"tasks={N} wins={wins}/{len(d)} gmean ratio={ratio:.3f} beats SNaive: {beats}")
print(sk.round(1).to_string())
