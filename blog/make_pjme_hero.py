"""Timestral-only fan chart on PJM East hourly load. Run: .venv\\Scripts\\python.exe blog\\make_pjme_hero.py"""
import sys
from pathlib import Path

import kagglehub
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from timestral import TimestralForecaster

sys.path.insert(0, str(Path(__file__).parent))
from style import BG, GRID, INK, INK2, TIMESTRAL  # noqa: E402

OUT = Path(__file__).parent
START, H = pd.Timestamp("2015-02-02"), 24 * 7   # a typical week: its error is close to the 2015 median

csv = Path(kagglehub.dataset_download("robikscube/hourly-energy-consumption")) / "PJME_hourly.csv"
pjme = pd.read_csv(csv, index_col=0, parse_dates=[0])["PJME_MW"].sort_index()
pjme = pjme[~pjme.index.duplicated()].asfreq("h").interpolate() / 1000   # GW

hist = pjme[:START - pd.Timedelta(hours=1)]
fut = pjme[START:START + pd.Timedelta(hours=H - 1)]
fc = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cpu").forecast(
    hist.iloc[-24 * 7 * 52:].to_numpy(np.float32), prediction_length=H)
q = lambda p: fc.quantile(p)[0, 0].cpu().numpy()  # noqa: E731
med = q(0.5)
mape = np.mean(np.abs(fut.to_numpy() - med) / fut.to_numpy()) * 100

show = hist.iloc[-24 * 7:]
fig, ax = plt.subplots(figsize=(16, 6.4))
ax.plot(show.index, show, color=INK2, lw=1.6)
for lo, hi, alpha, label in [(0.01, 0.99, 0.12, "98%"), (0.1, 0.9, 0.22, "80%"), (0.25, 0.75, 0.38, "50%")]:
    ax.fill_between(fut.index, q(lo), q(hi), color=TIMESTRAL, alpha=alpha, lw=0, label=f"{label} interval")
ax.plot(fut.index, med, color=TIMESTRAL, lw=2.6, label="Timestral forecast")
ax.scatter(fut.index, fut, s=9, color=INK, zorder=4, label="actual")
ax.axvline(START, color=INK2, lw=1, ls="--")
ax.annotate("forecast made here", (START, ax.get_ylim()[1]), xytext=(-8, -4), textcoords="offset points",
            ha="right", va="top", fontsize=11.5, color=INK2)
ax.yaxis.grid(True, color=GRID, lw=1)
ax.set_axisbelow(True)
ax.set_ylabel("load (GW)")
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
ax.legend(loc="lower left", frameon=False, fontsize=11.5, ncol=5)
fig.suptitle("One model, no training: next week's electricity demand, hour by hour", x=0.012, ha="left",
             fontsize=19, fontweight="bold", y=0.975)
fig.text(0.012, 0.875, f"PJM East hourly load. Timestral-209M, zero-shot on CPU, given one year of history. "
         f"Error on this week: {mape:.1f}% (MAPE), a typical week for 2015.", fontsize=12.5, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.87))
fig.savefig(OUT / "fig6_pjme_timestral.png")
print(f"MAPE {mape:.2f}%")
