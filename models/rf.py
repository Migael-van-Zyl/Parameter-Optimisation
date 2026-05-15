

import pandas as pd
import numpy as np
import os
from itertools import product

from sklearn.ensemble import RandomForestRegressor

# =========================
# SETTINGS
# =========================
file_path = r"C:\Users\MauduH\OneDrive - Eskom Holdings SOC Ltd\Python Projects\Parameter-Optimisation\data\Input Dataset - cleanedFortrack_dataset.xlsx"
output_folder = r"C:\Users\MauduH\OneDrive - Eskom Holdings SOC Ltd\Python Projects\Parameter-Optimisation\RandomForest_Search_Outputs"

os.makedirs(output_folder, exist_ok=True)

category = "Agriculture"
subsic = "Berries" 

consumption_col = "totalconsumption"

max_customers = 10
min_history = 24
lags = 3

# =========================
# METRICS
# =========================
def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0

    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean(np.abs(y_true - y_pred))


# =========================
# LOAD DATA
# =========================
df = pd.read_excel(file_path)

df["reportingmonth"] = pd.to_datetime(df["reportingmonth"], errors="coerce")

df = df.dropna(subset=["reportingmonth", "category", "subsic", "customerid"])

df = df[
    (df["category"] == category) &
    (df["subsic"] == subsic)
].copy()

print("\n==============================")
print("RANDOM FOREST PARAMETER SEARCH")
print("==============================")
print("Category:", category)
print("SubSIC:", subsic)
print("Filtered rows:", len(df))
print("Unique customers:", df["customerid"].nunique())


# =========================
# HELPER FUNCTIONS
# =========================
def prepare_customer_series(dataframe, customer_id):
    customer_df = dataframe[dataframe["customerid"] == customer_id].copy()

    ts = (
        customer_df
        .groupby("reportingmonth")[consumption_col]
        .sum()
        .sort_index()
    )

    return ts


def create_lags(ts, lags=3):
    lag_df = pd.DataFrame({"target": ts})

    for i in range(1, lags + 1):
        lag_df[f"lag_{i}"] = lag_df["target"].shift(i)

    return lag_df.dropna()


def train_test_split_data(lag_df, train_ratio=0.8):
    split = int(len(lag_df) * train_ratio)

    train = lag_df.iloc[:split]
    test = lag_df.iloc[split:]

    X_train = train.drop(columns=["target"])
    y_train = train["target"]

    X_test = test.drop(columns=["target"])
    y_test = test["target"]

    return X_train, X_test, y_train, y_test


def get_series_stats(ts):
    return {
        "records": len(ts),
        "mean": ts.mean(),
        "std": ts.std(),
        "min": ts.min(),
        "max": ts.max(),
        "zero_count": (ts == 0).sum(),
        "negative_count": (ts < 0).sum()
    }


# =========================
#RANDOM FOREST PARAMETER GRID
# =========================
n_estimators_values = [100, 200, 300, 400, 500]
max_depth_values = [5, 10, 15]
min_samples_split_values = [2, 5]

total_combinations = (
    len(n_estimators_values)
    * len(max_depth_values)
    * len(min_samples_split_values)
)

print("\nParameter combinations per customer:", total_combinations)

results = []

customers = df["customerid"].dropna().unique()[:max_customers]

# =========================
# LOOP THROUGH CUSTOMERS
# =========================
for i, customer in enumerate(customers, start=1):

    print(f"\nProcessing customer {i}/{len(customers)}: {customer}")

    ts = prepare_customer_series(df, customer)

    if len(ts) < min_history:
        print(f"Skipped customer {customer}: only {len(ts)} records")
        continue

    stats = get_series_stats(ts)

    lag_df = create_lags(ts, lags=lags)

    if len(lag_df) < 10:
        print(f"Skipped customer {customer}: not enough lagged records")
        continue

    X_train, X_test, y_train, y_test = train_test_split_data(lag_df)

    best_mape = np.inf
    best_mae = np.inf
    best_params = None

    checked = 0

    for n_estimators, max_depth, min_samples_split in product(
        n_estimators_values,
        max_depth_values,
        min_samples_split_values
    ):

        checked += 1

        try:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            score_mape = mape(y_test, preds)
            score_mae = mae(y_test, preds)

            if np.isfinite(score_mape) and score_mape < best_mape:
                best_mape = score_mape
                best_mae = score_mae
                best_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split
                }

        except Exception as e:
            print("Failed parameter combination:", e)
            continue

    if best_params is not None:
        results.append({
            "category": category,
            "subsic": subsic,
            "customerid": customer,
            "best_n_estimators": best_params["n_estimators"],
            "best_max_depth": best_params["max_depth"],
            "best_min_samples_split": best_params["min_samples_split"],
            "best_MAPE": best_mape,
            "best_MAE": best_mae,
            **stats
        })

        print(
            f"Best RF for customer {customer}: "
            f"{best_params} | MAPE: {best_mape:.2f}% | MAE: {best_mae:.2f}"
        )

    else:
        print(f"No valid Random Forest model found for customer {customer}")


# =========================
# EXPORT RESULTS
# =========================
results_df = pd.DataFrame(results)

safe_category = category.lower().replace(" ", "_")
safe_subsic = subsic.lower().replace(" ", "_")

best_file = os.path.join(
    output_folder,
    f"{safe_category}_{safe_subsic}_best_random_forest_results.xlsx"
)

freq_file = os.path.join(
    output_folder,
    f"{safe_category}_{safe_subsic}_random_forest_parameter_frequency.xlsx"
)

results_df.to_excel(best_file, index=False)

if not results_df.empty:
    param_frequency = (
        results_df
        .groupby([
            "category",
            "subsic",
            "best_n_estimators",
            "best_max_depth",
            "best_min_samples_split"
        ])
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
    )

    param_frequency.to_excel(freq_file, index=False)

    print("\nMost common Random Forest parameters:")
    print(param_frequency.head())

print("\nRandom Forest processing complete.")
print("Best results saved to:", best_file)
print("Parameter frequency saved to:", freq_file)