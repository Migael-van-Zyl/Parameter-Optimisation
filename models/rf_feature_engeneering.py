import pandas as pd
import numpy as np
import os
from itertools import product
from sklearn.ensemble import RandomForestRegressor

# =========================
# SETTINGS
# =========================
file_path = r"C:\Users\MauduH\OneDrive - Eskom Holdings SOC Ltd\Python Projects\Parameter-Optimisation\data\Input Dataset - cleanedFortrack_dataset.xlsx"


output_folder = r"C:\Users\MauduH\OneDrive - Eskom Holdings SOC Ltd\Python Projects\Parameter-Optimisation\results\rf_feature_engineering_results"
os.makedirs(output_folder, exist_ok=True)

category = "Agriculture"
subsic = "Cereal"   

consumption_col = "totalconsumption"

max_customers = 10
min_history = 24

lags = list(range(12, 25)) + [36, 48]

n_estimators_values = [100, 200, 300, 400, 500]
max_depth_values = [5, 10, 15]
min_samples_split_values = [2, 5]

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
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


# =========================
# FEATURE FUNCTIONS
# =========================
def season_from_month(month):
    if month in [12, 1, 2]:
        return "Summer"
    elif month in [3, 4, 5]:
        return "Autumn"
    elif month in [6, 7, 8]:
        return "Winter"
    else:
        return "Spring"


def demand_season_from_month(month):
    if month in [6, 7, 8]:
        return "High"
    else:
        return "Low"


def create_features(ts, lags):
    df_feat = pd.DataFrame({"target": ts})

    df_feat["month"] = df_feat.index.month

    # Holiday count feature
    holiday_map = {
        1: 1,
        2: 0,
        3: 1,
        4: 4,
        5: 1,
        6: 1,
        7: 0,
        8: 1,
        9: 1,
        10: 0,
        11: 0,
        12: 3
    }

    df_feat["holiday_count"] = df_feat["month"].map(holiday_map)

    # Cyclical month features
    df_feat["month_sin"] = np.sin(2 * np.pi * df_feat["month"] / 12)
    df_feat["month_cos"] = np.cos(2 * np.pi * df_feat["month"] / 12)

    # Season and demand features
    df_feat["season"] = df_feat["month"].apply(season_from_month)
    df_feat["demand_season"] = df_feat["month"].apply(demand_season_from_month)

    df_feat = pd.get_dummies(
        df_feat,
        columns=["season", "demand_season"],
        drop_first=False
    )

    # Lag features
    for lag in lags:
        df_feat[f"lag_{lag}"] = df_feat["target"].shift(lag)

    return df_feat.dropna()


def prepare_customer_series(dataframe, customer_id):
    customer_df = dataframe[dataframe["customerid"] == customer_id].copy()

    ts = (
        customer_df
        .groupby("reportingmonth")[consumption_col]
        .sum()
        .sort_index()
    )

    return ts


def train_test_split_data(feature_df, train_ratio=0.8):
    split = int(len(feature_df) * train_ratio)

    train = feature_df.iloc[:split]
    test = feature_df.iloc[split:]

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
print("RANDOM FOREST WITH FEATURE ENGINEERING")
print("==============================")
print("Category:", category)
print("SubSIC:", subsic)
print("Rows:", len(df))
print("Unique customers:", df["customerid"].nunique())
print("Lags used:", lags)

# =========================
# LOOP THROUGH CUSTOMERS
# =========================
results = []

customers = df["customerid"].dropna().unique()[:max_customers]

for i, customer in enumerate(customers, start=1):
    print(f"\nProcessing customer {i}/{len(customers)}: {customer}")

    ts = prepare_customer_series(df, customer)

    if len(ts) < min_history:
        print(f"Skipped customer {customer}: only {len(ts)} records")
        continue

    stats = get_series_stats(ts)

    feature_df = create_features(ts, lags)

    if len(feature_df) < 10:
        print(f"Skipped customer {customer}: not enough rows after creating lags")
        continue

    X_train, X_test, y_train, y_test = train_test_split_data(feature_df)

    best_mape = np.inf
    best_mae = np.inf
    best_params = None

    for n_estimators, max_depth, min_samples_split in product(
        n_estimators_values,
        max_depth_values,
        min_samples_split_values
    ):
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
            print("Failed combination:", e)
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
            "lags_used": str(lags),
            "features_used": "lags, month_sin, month_cos, season, demand_season, holiday_count",
            **stats
        })

        print(
            f"Best RF for customer {customer}: {best_params} | "
            f"MAPE: {best_mape:.2f}% | MAE: {best_mae:.2f}"
        )

# =========================
# EXPORT RESULTS
# =========================
results_df = pd.DataFrame(results)

safe_category = category.lower().replace(" ", "_")
safe_subsic = subsic.lower().replace(" ", "_")

best_file = os.path.join(
    output_folder,
    f"{safe_category}_{safe_subsic}_best_rf_feature_engineering_results.xlsx"
)

freq_file = os.path.join(
    output_folder,
    f"{safe_category}_{safe_subsic}_rf_feature_engineering_parameter_frequency.xlsx"
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

    print("\nMost common RF parameters:")
    print(param_frequency.head())

print("\nDone.")
print("Best results saved to:", best_file)
print("Parameter frequency saved to:", freq_file)