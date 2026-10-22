"""Rolling week-ahead backtest on PJM East hourly load, every week of 2015, Prophet vs Timestral.

Each week is cached in blog/pjme_backtest.csv so the run can be resumed.
Run: .venv\\Scripts\\python.exe blog\\pjme_backtest.py
"""
import logging
import time
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from prophet import Prophet
from timestral import TimestralForecaster

for _name in ("cmdstanpy", "prophet"):
    logging.getLogger(_name).disabled = True

OUT = Path(__file__).parent / "pjme_backtest.csv"
H = 24 * 7
TIMESTRAL_CONTEXT = 24 * 7 * 52      # one year
PROPHET_TRAIN = 24 * 365 * 3         # three years keeps each fit fast

csv = Path(kagglehub.dataset_download("robikscube/hourly-energy-consumption")) / "PJME_hourly.csv"
pjme = pd.read_csv(csv, index_col=0, parse_dates=[0])["PJME_MW"].sort_index()
pjme = pjme[~pjme.index.duplicated()].asfreq("h").interpolate()
forecaster = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cpu")

done = pd.read_csv(OUT, parse_dates=["week"]) if OUT.exists() else pd.DataFrame()
starts = pd.date_range("2015-01-05", "2015-12-28", freq="W-MON")  # Monday 00:00 each week


def scores(y, med, lo, hi):
    return {"mape": float(np.mean(np.abs(y - med) / y) * 100), "coverage": float(np.mean((y >= lo) & (y <= hi)) * 100)}


for start in starts:
    if len(done) and (done.week == start).any():
        continue
    t0 = time.time()
    hist = pjme[:start - pd.Timedelta(hours=1)]
    y = pjme[start:start + pd.Timedelta(hours=H - 1)].to_numpy()

    fc = forecaster.forecast(hist.iloc[-TIMESTRAL_CONTEXT:].to_numpy(np.float32), prediction_length=H)
    q = lambda p: fc.quantile(p)[0, 0].cpu().numpy()  # noqa: E731
    ts = scores(y, q(0.5), q(0.1), q(0.9))

    m = Prophet(interval_width=0.8).fit(hist.iloc[-PROPHET_TRAIN:].rename("y").rename_axis("ds").reset_index())
    p = m.predict(pd.DataFrame({"ds": pd.date_range(start, periods=H, freq="h")}))
    ps = scores(y, p.yhat.to_numpy(), p.yhat_lower.to_numpy(), p.yhat_upper.to_numpy())

    row = {"week": start, "timestral_mape": ts["mape"], "timestral_coverage": ts["coverage"],
           "prophet_mape": ps["mape"], "prophet_coverage": ps["coverage"]}
    done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
    done.to_csv(OUT, index=False)
    print(f"{start:%Y-%m-%d}  Timestral {ts['mape']:5.1f}%  Prophet {ps['mape']:5.1f}%  ({time.time() - t0:.0f}s)",
          flush=True)
