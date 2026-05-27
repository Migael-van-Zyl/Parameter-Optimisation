import pandas as pd
import numpy as np
import os
from itertools import product
from sklearn.ensemble import RandomForestRegressor

# =========================
# SETTINGS
# =========================
file_path = r"C:\Users\MauduH\OneDrive - Eskom Holdings SOC Ltd\Python Projects\Parameter-Optimisation\data\Input Dataset - cleanedFortrack_dataset.xlsx"
output_folder = r"C:\Users\MauduH\OneDrive - Eskom Holdings SOC Ltd\Python Projects\Parameter-Optimisation\RandomForest_Outputs"
os.makedirs(output_folder, exist_ok=True)

consumption_columns = [
    "offpeakconsumption",
    "standardconsumption",
    "peakconsumption",
    "totalconsumption"
]

min_history = 24

max_customers_per_subsic = 20

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
# FEATURE ENGINEERING
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

    df_feat["month_sin"] = np.sin(2 * np.pi * df_feat["month"] / 12)
    df_feat["month_cos"] = np.cos(2 * np.pi * df_feat["month"] / 12)

    df_feat["season"] = df_feat["month"].apply(season_from_month)
    df_feat["demand_season"] = df_feat["month"].apply(demand_season_from_month)

    df_feat = pd.get_dummies(
        df_feat,
        columns=["season", "demand_season"],
        drop_first=False
    )

    for lag in lags:
        df_feat[f"lag_{lag}"] = df_feat["target"].shift(lag)

    return df_feat.dropna()


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

df = df.dropna(
    subset=[
        "reportingmonth",
        "category",
        "subsic",
        "customerid"
    ]
)

print("\n===================================")
print("RANDOM FOREST - ALL CATEGORIES + ALL CONSUMPTIONS")
print("===================================")
print("Total rows:", len(df))
print("Categories:", df["category"].nunique())
print("SubSICs:", df["subsic"].nunique())
print("Customers:", df["customerid"].nunique())
print("Consumption columns:", consumption_columns)

# =========================
# MASTER RESULT LISTS
# =========================
all_customer_results = []
all_parameter_library = []

# =========================
# LOOP CONSUMPTION → CATEGORY → SUBSIC → CUSTOMER
# =========================
for consumption_col in consumption_columns:

    if consumption_col not in df.columns:
        print(f"\nSkipping {consumption_col}: column not found")
        continue

    print("\n================================================")
    print("CONSUMPTION COLUMN:", consumption_col)
    print("================================================")

    df_consumption = df.dropna(subset=[consumption_col]).copy()

    for category in sorted(df_consumption["category"].dropna().unique()):

        category_df = df_consumption[df_consumption["category"] == category].copy()

        for subsic in sorted(category_df["subsic"].dropna().unique()):

            subsic_df = category_df[category_df["subsic"] == subsic].copy()

            customers = subsic_df["customerid"].dropna().unique()

            if max_customers_per_subsic is not None:
                customers = customers[:max_customers_per_subsic]

            print("\n-----------------------------------")
            print("Consumption:", consumption_col)
            print("Category:", category)
            print("SubSIC:", subsic)
            print("Customers to check:", len(customers))
            print("-----------------------------------")

            subsic_results = []

            for i, customer in enumerate(customers, start=1):

                print(f"Processing customer {i}/{len(customers)}: {customer}")

                customer_df = subsic_df[subsic_df["customerid"] == customer].copy()

                ts = (
                    customer_df
                    .groupby("reportingmonth")[consumption_col]
                    .sum()
                    .sort_index()
                )

                if len(ts) < min_history:
                    print(f"Skipped {customer}: only {len(ts)} records")
                    continue

                stats = get_series_stats(ts)

                feature_df = create_features(ts, lags)

                if len(feature_df) < 10:
                    print(f"Skipped {customer}: not enough rows after lags")
                    continue

                split = int(len(feature_df) * 0.8)

                train = feature_df.iloc[:split]
                test = feature_df.iloc[split:]

                X_train = train.drop(columns=["target"])
                y_train = train["target"]

                X_test = test.drop(columns=["target"])
                y_test = test["target"]

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

                    result = {
                        "consumption_column": consumption_col,
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
                    }

                    all_customer_results.append(result)
                    subsic_results.append(result)

                    print(
                        f"Best RF: {best_params} | "
                        f"MAPE: {best_mape:.2f}% | MAE: {best_mae:.2f}"
                    )

            # =========================
            # PARAMETER LIBRARY PER CONSUMPTION + SUBSIC
            # =========================
            if subsic_results:

                subsic_results_df = pd.DataFrame(subsic_results)

                param_frequency = (
                    subsic_results_df
                    .groupby([
                        "consumption_column",
                        "category",
                        "subsic",
                        "best_n_estimators",
                        "best_max_depth",
                        "best_min_samples_split"
                    ])
                    .agg(
                        count=("customerid", "count"),
                        avg_MAPE=("best_MAPE", "mean"),
                        avg_MAE=("best_MAE", "mean")
                    )
                    .reset_index()
                    .sort_values(
                        by=["count", "avg_MAPE"],
                        ascending=[False, True]
                    )
                )

                top_row = param_frequency.iloc[0]

                all_parameter_library.append({
                    "consumption_column": consumption_col,
                    "category": category,
                    "subsic": subsic,
                    "model": "Random Forest",
                    "recommended_n_estimators": top_row["best_n_estimators"],
                    "recommended_max_depth": top_row["best_max_depth"],
                    "recommended_min_samples_split": top_row["best_min_samples_split"],
                    "parameter_count": top_row["count"],
                    "avg_MAPE": top_row["avg_MAPE"],
                    "avg_MAE": top_row["avg_MAE"],
                    "valid_customer_count": len(subsic_results_df)
                })

# =========================
# EXPORT RESULTS
# =========================
customer_results_df = pd.DataFrame(all_customer_results)
parameter_library_df = pd.DataFrame(all_parameter_library)

customer_results_file = os.path.join(
    output_folder,
    "all_categories_all_consumptions_rf_customer_best_results.xlsx"
)

parameter_library_file = os.path.join(
    output_folder,
    "all_categories_all_consumptions_rf_parameter_library.xlsx"
)

customer_results_df.to_excel(customer_results_file, index=False)
parameter_library_df.to_excel(parameter_library_file, index=False)

print("\n===================================")
print("RANDOM FOREST ALL CATEGORIES + ALL CONSUMPTIONS DONE")
print("Customer results saved to:", customer_results_file)
print("Parameter library saved to:", parameter_library_file)
print("===================================")