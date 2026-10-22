"""Week-by-week error over 2015 from blog/pjme_backtest.csv. Run: .venv\\Scripts\\python.exe blog\\make_pjme_year_figure.py"""
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from style import BG, GRID, INK2, PROPHET, TIMESTRAL  # noqa: E402

OUT = Path(__file__).parent
df = pd.read_csv(OUT / "pjme_backtest.csv", parse_dates=["week"]).sort_values("week")
n = len(df)
wins = int((df.timestral_mape < df.prophet_mape).sum())
med_t, med_p = df.timestral_mape.median(), df.prophet_mape.median()
cov_t, cov_p = df.timestral_coverage.mean(), df.prophet_coverage.mean()

fig, ax = plt.subplots(figsize=(14, 6.4))
ax.fill_between(df.week, df.timestral_mape, df.prophet_mape, where=df.prophet_mape >= df.timestral_mape,
                color=TIMESTRAL, alpha=0.10, lw=0, interpolate=True)
ax.fill_between(df.week, df.timestral_mape, df.prophet_mape, where=df.prophet_mape < df.timestral_mape,
                color=PROPHET, alpha=0.15, lw=0, interpolate=True)
last_p, last_t = df.prophet_mape.iloc[-1], df.timestral_mape.iloc[-1]
nudge = max(0.0, 1.2 - abs(last_p - last_t)) / 2  # keep the two end labels apart
for col, color, name, dy in [("prophet_mape", PROPHET, "Prophet", nudge if last_p >= last_t else -nudge),
                             ("timestral_mape", TIMESTRAL, "Timestral", -nudge if last_p >= last_t else nudge)]:
    ax.plot(df.week, df[col], color=color, lw=2.4, marker="o", ms=5, mec=BG, mew=1.5, label=name)
    ax.annotate(name, (df.week.iloc[-1], df[col].iloc[-1] + dy), xytext=(10, 0), textcoords="offset points",
                color=color, fontsize=14, fontweight="bold", va="center")
ax.set_ylim(0, None)
ax.set_xlim(df.week.iloc[0] - pd.Timedelta(days=4), df.week.iloc[-1] + pd.Timedelta(days=4))
ax.yaxis.grid(True, color=GRID, lw=1)
ax.set_axisbelow(True)
ax.yaxis.set_major_locator(plt.MultipleLocator(5))
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.set_ylabel("week-ahead error (MAPE)")
ax.legend(loc="upper left", frameon=False, fontsize=12, ncol=2)
fig.suptitle(f"A full year of week-ahead forecasts: Timestral wins {wins} of {n} weeks", x=0.015, ha="left",
             fontsize=19, fontweight="bold", y=0.975)
fig.text(0.015, 0.83,
         f"PJM East hourly electricity load, 2015. Each Monday both models forecast the next 168 hours.\n"
         f"Median weekly error: Timestral {med_t:.1f}% vs Prophet {med_p:.1f}%. "
         f"80% intervals contain {cov_t:.0f}% (Timestral) vs {cov_p:.0f}% (Prophet) of actual values.",
         fontsize=12.5, color=INK2, linespacing=1.5)
fig.tight_layout(rect=(0, 0, 0.96, 0.82))
fig.savefig(OUT / "fig5_pjme_year.png")
print(f"{n} weeks, Timestral wins {wins}, median MAPE {med_t:.2f} vs {med_p:.2f}, coverage {cov_t:.0f} vs {cov_p:.0f}")
