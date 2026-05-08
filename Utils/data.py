# import pandas as pd


# # ----------------------------
# # LOAD DATASET
# # ----------------------------
# def load_dataset(base_dir, relative_path="data/cleanedFortrack_dataset.xlsx"):
#     file_path = base_dir / relative_path
#     df = pd.read_excel(file_path)
#     df["reportingmonth"] = pd.to_datetime(df["reportingmonth"])
#     return df


# # ----------------------------
# # FILTER DATA
# # ----------------------------
# def filter_data(df, category=None, subsic=None, customer_id=None):
#     data = df.copy()

#     if category is not None:
#         data = data[data["category"] == category]

#     if subsic is not None:
#         data = data[data["subsic"] == subsic]

#     if customer_id is not None:
#         data = data[data["customerid"] == customer_id]

#     return data


# # ----------------------------
# # CREATE MONTHLY SERIES
# # ----------------------------
# def make_customer_monthly_series(customer_df, target_col="totalconsumption"):
#     ts = (
#         customer_df
#         .groupby("reportingmonth")[target_col]
#         .sum()
#         .sort_index()
#     )

#     ts.index = pd.to_datetime(ts.index).to_period("M").to_timestamp()
#     return ts


# # ----------------------------
# # TRAIN / TEST SPLIT (TIME SERIES)
# # ----------------------------
# def split_series_time(ts, test_size=0.2):
#     split_idx = int(len(ts) * (1 - test_size))
#     train = ts.iloc[:split_idx]
#     test = ts.iloc[split_idx:]
#     return train, test


# # ----------------------------
# # CREATE LAGGED DATA
# # ----------------------------
# def make_lagged_frame(ts, lags=3):
#     df = pd.DataFrame({"target": ts})

#     for i in range(1, lags + 1):
#         df[f"lag_{i}"] = df["target"].shift(i)

#     df = df.dropna()
#     return df

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