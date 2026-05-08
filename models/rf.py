

# from itertools import product
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor

# from Utils.data import make_customer_monthly_series, split_series_time, create_lagged_frame
# from Utils.metrics import mape


# def run_random_forest_parameter_search(
#     customer_df: pd.DataFrame,
#     customer_id,
#     lags: int = 3,
#     param_grid: dict | None = None,
# ):
#     if param_grid is None:
#         param_grid = {
#             "n_estimators": [100, 200],
#             "max_depth": [5, 10],
#             "min_samples_split": [2, 5],
#         }

#     ts = make_customer_monthly_series(customer_df)

#     if len(ts) < 12:
#         return None

#     lagged = create_lagged_frame(ts, lags=lags)
#     if len(lagged) < 8:
#         return None

#     train_df, test_df = split_series_time(lagged, test_size=0.2)

#     X_train = train_df.drop(columns=["target"])
#     y_train = train_df["target"]

#     X_test = test_df.drop(columns=["target"])
#     y_test = test_df["target"]

#     best_mape = float("inf")
#     best_params = None

#     for n_estimators, max_depth, min_samples_split in product(
#         param_grid["n_estimators"],
#         param_grid["max_depth"],
#         param_grid["min_samples_split"],
#     ):
#         model = RandomForestRegressor(
#             n_estimators=n_estimators,
#             max_depth=max_depth,
#             min_samples_split=min_samples_split,
#             random_state=42,
#         )

#         model.fit(X_train, y_train)
#         preds = model.predict(X_test)
#         score = mape(y_test, preds)

#         if score < best_mape:
#             best_mape = score
#             best_params = {
#                 "n_estimators": n_estimators,
#                 "max_depth": max_depth,
#                 "min_samples_split": min_samples_split,
#             }

#     return {
#         "customerid": customer_id,
#         "best_params": best_params,
#         "mape": best_mape,
#     }

import pandas as pd


# =========================
# LOAD DATA
# =========================
def load_dataset(base_dir, relative_path="data/cleanedFortrack_dataset.xlsx"):
    file_path = base_dir / relative_path
    df = pd.read_excel(file_path)
    df["reportingmonth"] = pd.to_datetime(df["reportingmonth"])
    return df


# =========================
# FILTER DATA
# =========================
def filter_data(df, category=None, subsic=None, customer_id=None):

    data = df.copy()

    if category is not None:
        data = data[data["category"] == category]

    if subsic is not None:
        data = data[data["subsic"] == subsic]

    if customer_id is not None:
        data = data[data["customerid"] == customer_id]

    return data


# =========================
# CREATE MONTHLY SERIES
# =========================
def make_customer_monthly_series(customer_df, target_col="totalconsumption"):

    ts = (
        customer_df
        .groupby("reportingmonth")[target_col]
        .sum()
        .sort_index()
    )

    ts.index = pd.to_datetime(ts.index).to_period("M").to_timestamp()

    return ts


# =========================
# SPLIT TIME SERIES
# =========================
def split_series_time(ts, test_size=0.2):

    split_idx = int(len(ts) * (1 - test_size))

    train = ts.iloc[:split_idx]
    test = ts.iloc[split_idx:]

    return train, test


# =========================
# CREATE LAGGED FEATURES
# =========================
def create_lagged_frame(ts, lags=3):

    df = pd.DataFrame({"target": ts})

    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["target"].shift(i)

    df = df.dropna()

    return df