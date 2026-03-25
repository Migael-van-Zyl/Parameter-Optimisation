
from itertools import product
import pandas as pd

from Utils.data import make_customer_monthly_series, split_series_time, create_lagged_frame
from Utils.metrics import mape

try:
    from xgboost import XGBRegressor
except ImportError as e:
    raise ImportError("xgboost is not installed. Run: pip install xgboost") from e


def run_xgboost_parameter_search(
    customer_df: pd.DataFrame,
    customer_id,
    lags: int = 3,
    param_grid: dict | None = None,
):
    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "learning_rate": [0.05, 0.1],
        }

    ts = make_customer_monthly_series(customer_df)

    if len(ts) < 12:
        return None

    lagged = create_lagged_frame(ts, lags=lags)
    if len(lagged) < 8:
        return None

    train_df, test_df = split_series_time(lagged, test_size=0.2)

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    best_mape = float("inf")
    best_params = None

    for n_estimators, max_depth, learning_rate in product(
        param_grid["n_estimators"],
        param_grid["max_depth"],
        param_grid["learning_rate"],
    ):
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="reg:squarederror",
            random_state=42,
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = mape(y_test, preds)

        if score < best_mape:
            best_mape = score
            best_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
            }

    return {
        "customerid": customer_id,
        "best_params": best_params,
        "mape": best_mape,
    }