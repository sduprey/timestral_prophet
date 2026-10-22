"""Side-by-side forecast showcases, Prophet vs Timestral. Run: .venv\\Scripts\\python.exe blog\\make_forecast_figures.py"""
import logging
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from timestral import TimestralForecaster

sys.path.insert(0, str(Path(__file__).parent))
from style import BG, GRID, INK, INK2, PROPHET, TIMESTRAL  # noqa: E402

for _name in ("cmdstanpy", "prophet"):
    logging.getLogger(_name).disabled = True

OUT = Path(__file__).parent
MODEL_ID = "mistralai/Timestral-209M"
forecaster = TimestralForecaster.from_pretrained(MODEL_ID, device="cpu")


def mape(y, yhat):
    return float(np.mean(np.abs((y - yhat) / y)) * 100)


def timestral_fc(history, horizon):
    """Median and 80% interval."""
    fc = forecaster.forecast(np.asarray(history, dtype=np.float32), prediction_length=horizon)
    q = lambda p: fc.quantile(p)[0, 0].cpu().numpy()  # noqa: E731
    return q(0.5), q(0.1), q(0.9)


def coverage(y, lo, hi):
    return float(np.mean((y >= lo) & (y <= hi)) * 100)


def prophet_fc(train: pd.DataFrame, future_ds: pd.Series, **kw):
    """Prophet (default settings unless overridden): median and 80% interval."""
    m = Prophet(interval_width=0.8, **kw)
    m.fit(train)
    p = m.predict(pd.DataFrame({"ds": future_ds}))
    return p["yhat"].to_numpy(), p["yhat_lower"].to_numpy(), p["yhat_upper"].to_numpy()


def duel(hist_x, hist_y, fut_x, fut_y, fcs, title, subtitle, ylabel, fname, figsize, date_fmt=None, stacked=False):
    """One panel per model, same axes: history, 80% band, median, actuals."""
    n = len(fcs)
    fig, axes = plt.subplots(n if stacked else 1, 1 if stacked else n, figsize=figsize, sharey=True,
                             sharex=stacked)
    for ax, (name, color, (med, lo, hi)) in zip(np.atleast_1d(axes), fcs):
        ax.plot(hist_x, hist_y, color=INK2, lw=1.3)
        ax.fill_between(fut_x, lo, hi, color=color, alpha=0.18, lw=0, label="80% interval")
        ax.plot(fut_x, fut_y, color=INK, lw=1.6, label="actual")
        ax.plot(fut_x, med, color=color, lw=2.4, label="forecast")
        ax.axvline(fut_x[0], color=GRID, lw=1.5, ls="--", zorder=0)
        ax.yaxis.grid(True, color=GRID, lw=1)
        ax.set_axisbelow(True)
        sep = "   ·   " if stacked else "\n"
        ax.set_title(f"{name}   ·   error (MAPE) {mape(fut_y, med):.1f}%{sep}"
                     f"80% interval holds {coverage(fut_y, lo, hi):.0f}% of actuals", loc="left", fontsize=14,
                     fontweight="bold", color=color, linespacing=1.5)
        if date_fmt:
            ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
        ax.legend(loc="upper left", frameon=False, fontsize=11, ncol=3)
    np.atleast_1d(axes)[0].set_ylabel(ylabel)
    fig.suptitle(title, x=0.015, ha="left", fontsize=19, fontweight="bold", y=0.985)
    fig.text(0.015, 0.905 if not stacked else 0.935, subtitle, fontsize=12.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.92 if not stacked else 0.945))
    fig.savefig(OUT / fname)
    print(fname, {name: round(mape(fut_y, f[0]), 2) for name, _, f in fcs})


# 1. Airline passengers, last 24 months held out (the classic Prophet demo series)
air = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv")
air["ds"] = pd.to_datetime(air["Month"])
air = air.rename(columns={"Passengers": "y"})[["ds", "y"]]
H = 24
train, test = air.iloc[:-H], air.iloc[-H:]
duel(train.ds, train.y, test.ds.to_numpy(), test.y.to_numpy(),
     [("Prophet", PROPHET, prophet_fc(train, test.ds, seasonality_mode="multiplicative")),
      ("Timestral", TIMESTRAL, timestral_fc(train.y, H))],
     "Prophet's home turf: sharper median, overconfident intervals",
     "Monthly airline passengers (k), last 2 years hidden. Prophet set up as in its docs (multiplicative "
     "seasonality); Timestral untouched.", "passengers (k)", "fig3_airline.png", figsize=(15, 5.8),
     date_fmt="%Y")


# 2. PJM East hourly load: week-ahead forecast for the week of July 4th 2015
def find_pjme() -> Path:
    import kagglehub
    return Path(kagglehub.dataset_download("robikscube/hourly-energy-consumption")) / "PJME_hourly.csv"


pjme = pd.read_csv(find_pjme(), index_col=0, parse_dates=[0])["PJME_MW"].sort_index()
pjme = pjme[~pjme.index.duplicated()].asfreq("h").interpolate()
start, H = pd.Timestamp("2015-07-01"), 24 * 7
hist = pjme[:start - pd.Timedelta(hours=1)]
fut = pjme[start:start + pd.Timedelta(hours=H - 1)]
context = hist.iloc[-24 * 7 * 52:]                    # one year of context for Timestral
prophet_train = hist.iloc[-24 * 365 * 3:].rename("y").rename_axis("ds").reset_index()  # 3 years keeps the fit fast
show = hist.iloc[-24 * 5:]
duel(show.index, show.to_numpy() / 1000, fut.index, fut.to_numpy() / 1000,
     [("Prophet", PROPHET, tuple(a / 1000 for a in prophet_fc(prophet_train, fut.index.to_series()))),
      ("Timestral", TIMESTRAL, tuple(a / 1000 for a in timestral_fc(context.to_numpy(), H)))],
     "Week-ahead electricity demand, July 4th week",
     "PJM East hourly load (GW), forecast made on 1 July 2015 for the next 7 days. No holiday features.",
     "load (GW)", "fig4_energy_week.png", figsize=(14, 8.4), date_fmt="%a %d", stacked=True)
