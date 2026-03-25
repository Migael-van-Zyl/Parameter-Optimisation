# import pandas as pd

# def load_and_prepare_data(file_path):
#     df = pd.read_excel(file_path)
#     df["reportingmonth"] = pd.to_datetime(df["reportingmonth"])

#     ts = df.groupby("reportingmonth")["totalconsumption"].sum().reset_index()
#     ts = ts.sort_values("reportingmonth").set_index("reportingmonth")

#     return ts


# def create_lags(ts, lags=3):
#     df = ts.copy()
#     for i in range(1, lags + 1):
#         df[f"lag_{i}"] = df["totalconsumption"].shift(i)

#     return df.dropna()


# def train_test_split_ts(ts, split_ratio=0.8):
#     split = int(len(ts) * split_ratio)
#     train = ts.iloc[:split]
#     test = ts.iloc[split:]
#     return train, test

# from pathlib import Path
# import pandas as pd


# def load_dataset(base_dir: Path, relative_path: str = "data/cleanedFortrack_dataset.xlsx") -> pd.DataFrame:
#     file_path = base_dir / relative_path
#     df = pd.read_excel(file_path)
#     df["reportingmonth"] = pd.to_datetime(df["reportingmonth"])
#     return df


# def filter_data(
#     df: pd.DataFrame,
#     category: str | None = None,
#     subsic: str | None = None,
#     customer_id: str | int | None = None,
# ) -> pd.DataFrame:
#     out = df.copy()

#     if category is not None:
#         out = out[out["category"] == category]

#     if subsic is not None:
#         out = out[out["subsic"] == subsic]

#     if customer_id is not None:
#         out = out[out["customerid"] == customer_id]

#     return out


# def make_customer_monthly_series(customer_df: pd.DataFrame, target_col: str = "totalconsumption") -> pd.Series:
#     ts = (
#         customer_df.groupby("reportingmonth")[target_col]
#         .sum()
#         .sort_index()
#     )
#     ts.index = pd.to_datetime(ts.index).to_period("M").to_timestamp()
#     return ts


# def split_series_time(ts: pd.Series, test_size: float = 0.2):
#     split_idx = int(len(ts) * (1 - test_size))
#     train = ts.iloc[:split_idx]
#     test = ts.iloc[split_idx:]
#     return train, test


# def create_lagged_frame(ts: pd.Series, lags: int = 3) -> pd.DataFrame:
#     df = pd.DataFrame({"target": ts})
#     for i in range(1, lags + 1):
#         df[f"lag_{i}"] = df["target"].shift(i)
#     return df.dropna()

# # import pandas as pd


# # def load_dataset(base_dir, relative_path="data/cleanedFortrack_dataset.xlsx"):
# #     file_path = base_dir / relative_path
# #     df = pd.read_excel(file_path)
# #     df["reportingmonth"] = pd.to_datetime(df["reportingmonth"])
# #     return df


# # def filter_data(df, category=None, subsic=None, customer_id=None):

# #     data = df.copy()

# #     if category is not None:
# #         data = data[data["category"] == category]

# #     if subsic is not None:
# #         data = data[data["subsic"] == subsic]

# #     if customer_id is not None:
# #         data = data[data["customerid"] == customer_id]

# #     return data

# #     import Utils.data
# #     print(dir(Utils.data))

# import pandas as pd


# def load_dataset(base_dir, relative_path="data/cleanedFortrack_dataset.xlsx"):
#     file_path = base_dir / relative_path
#     df = pd.read_excel(file_path)
#     df["reportingmonth"] = pd.to_datetime(df["reportingmonth"])
#     return df


# def filter_data(df, category=None, subsic=None, customer_id=None):

#     data = df.copy()

#     if category is not None:
#         data = data[data["category"] == category]

#     if subsic is not None:
#         data = data[data["subsic"] == subsic]

#     if customer_id is not None:
#         data = data[data["customerid"] == customer_id]

#     return data


# def make_customer_monthly_series(customer_df, target_col="totalconsumption"):

#     ts = (
#         customer_df
#         .groupby("reportingmonth")[target_col]
#         .sum()
#         .sort_index()
#     )

#     ts.index = pd.to_datetime(ts.index).to_period("M").to_timestamp()

#     return ts


# def split_series_time(ts, test_size=0.2):

#     split_idx = int(len(ts) * (1 - test_size))

#     train = ts.iloc[:split_idx]
#     test = ts.iloc[split_idx:]

#     return train, test


# def create_lagged_frame(ts, lags=3):

#     df = pd.DataFrame({"target": ts})

#     for i in range(1, lags + 1):
#         df[f"lag_{i}"] = df["target"].shift(i)

#     return df.dropna()

import pandas as pd


# ----------------------------
# LOAD DATASET
# ----------------------------
def load_dataset(base_dir, relative_path="data/cleanedFortrack_dataset.xlsx"):

    file_path = base_dir / relative_path
    df = pd.read_excel(file_path)

    df["reportingmonth"] = pd.to_datetime(df["reportingmonth"])

    return df


# ----------------------------
# FILTER DATA
# ----------------------------
def filter_data(df, category=None, subsic=None, customer_id=None):

    data = df.copy()

    if category is not None:
        data = data[data["category"] == category]

    if subsic is not None:
        data = data[data["subsic"] == subsic]

    if customer_id is not None:
        data = data[data["customerid"] == customer_id]

    return data


# ----------------------------
# CREATE MONTHLY SERIES
# ----------------------------
def make_customer_monthly_series(customer_df, target_col="totalconsumption"):

    ts = (
        customer_df
        .groupby("reportingmonth")[target_col]
        .sum()
        .sort_index()
    )

    ts.index = pd.to_datetime(ts.index).to_period("M").to_timestamp()

    return ts


# ----------------------------
# TRAIN TEST SPLIT (TIME SERIES)
# ----------------------------
def split_series_time(ts, test_size=0.2):

    split_idx = int(len(ts) * (1 - test_size))

    train = ts.iloc[:split_idx]
    test = ts.iloc[split_idx:]

    return train, test


# ----------------------------
# CREATE LAGGED DATA
# ----------------------------
def make_lagged_frame(ts, lags=3):

    df = pd.DataFrame({"target": ts})

    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["target"].shift(i)

    df = df.dropna()

    return df