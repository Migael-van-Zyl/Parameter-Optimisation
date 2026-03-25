
from itertools import product
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from Utils.data import make_customer_monthly_series, split_series_time
from Utils.metrics import mape


def run_arima_parameter_search(
    customer_df: pd.DataFrame,
    customer_id,
    param_grid: dict | None = None,
):
    if param_grid is None:
        param_grid = {
            "p": [0, 1, 2],
            "d": [0, 1],
            "q": [0, 1, 2],
        }

    ts = make_customer_monthly_series(customer_df)

    if len(ts) < 12:
        return None

    train_ts, test_ts = split_series_time(ts, test_size=0.2)

    best_mape = float("inf")
    best_params = None

    for p, d, q in product(param_grid["p"], param_grid["d"], param_grid["q"]):
        try:
            model = ARIMA(train_ts, order=(p, d, q)).fit()
            preds = model.forecast(steps=len(test_ts))
            score = mape(test_ts, preds)

            if score < best_mape:
                best_mape = score
                best_params = {"p": p, "d": d, "q": q}
        except Exception:
            continue

    return {
        "customerid": customer_id,
        "best_params": best_params,
        "mape": best_mape,
    }


def run_sarima_parameter_search(
    customer_df: pd.DataFrame,
    customer_id,
    param_grid: dict | None = None,
):
    if param_grid is None:
        param_grid = {
            "p": [0, 1],
            "d": [0, 1],
            "q": [0, 1],
            "P": [0, 1],
            "D": [0, 1],
            "Q": [0, 1],
            "s": [12],
        }

    ts = make_customer_monthly_series(customer_df)

    if len(ts) < 24:
        return None

    train_ts, test_ts = split_series_time(ts, test_size=0.2)

    best_mape = float("inf")
    best_params = None

    for p, d, q, P, D, Q, s in product(
        param_grid["p"],
        param_grid["d"],
        param_grid["q"],
        param_grid["P"],
        param_grid["D"],
        param_grid["Q"],
        param_grid["s"],
    ):
        try:
            model = SARIMAX(
                train_ts,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)

            preds = model.forecast(steps=len(test_ts))
            score = mape(test_ts, preds)

            if score < best_mape:
                best_mape = score
                best_params = {
                    "p": p, "d": d, "q": q,
                    "P": P, "D": D, "Q": Q, "s": s,
                }
        except Exception:
            continue

    return {
        "customerid": customer_id,
        "best_params": best_params,
        "mape": best_mape,
    }